# TB CAD Reliability Layer — Implementation PRD

Prototype engineering doc for the paper at `ijacsa-paper.tex`. Scope: build the five reliability modules around an off-the-shelf TB detector, in 4 weeks.

---

## System design

### Pipeline (synchronous, one CXR in → one panel out)

```
CXR (DICOM/PNG)
   │
   ▼
[1] Quality Gate ──► reject / manual-review / warning flag
   │ (if pass)
   ▼
[2] TB Detector (Faster R-CNN on TBX11K)
   │   emits: image-level score s, List<(bbox, class, p)>, feature vector φ
   │
   ├──► [3a] Lung-Zone Map (torchxrayvision anatomical seg)
   │         per-bbox anatomical descriptor
   │
   ├──► [3b] Finding Classifier (Shenzhen-trained)
   │         per-bbox fine-grained label
   │
   ├──► [4] Deferral
   │    • temperature-scaled s
   │    • threshold-band selective prediction
   │    • Mahalanobis OOD on φ
   │    └─► {confident-pos, confident-neg, uncertain}
   │
   ├──► [5] Explanation
   │    • template per (finding, zone, p)
   │    • LLM smoothing on text-only structured list (image withheld)
   │    • RadGraph-style NER verify; fallback to raw templates on failure
   │
   └──► [6] UI panel (React or Streamlit)
         + log event to [7] Monitoring store (Postgres + Parquet)
```

Everything above the UI runs in one Python process with a simple FastAPI endpoint; monitoring is a decoupled batch job over the event log.

---

## [1] TB Detector

### Architecture
- **Faster R-CNN, ResNet-50-FPN backbone.** Reference impl: `yun-liu/Tuberculosis` (PyTorch, mmdetection-style configs). Also exports RetinaNet, SSD, FCOS, SymFormer for ablation.
- Train schedule from paper: SGD, lr 0.02, batch 16 over 8 GPUs (scale linearly for single-GPU), 12 epochs, standard COCO augmentations + CLAHE pre-processing on CXR.
- **Output head:** 2 foreground classes (`active_tb`, `latent_tb`) + background. Image-level TB score = max over `active_tb` box scores (or a calibrated aggregation — see §4).
- **Feature vector φ for OOD:** pooled FPN features at the RoI head input. Cache φ per inference call; needed by the Mahalanobis stage.

### Paper + repo
- Liu et al., CVPR 2020 — TBX11K + baselines.
- Liu et al., TPAMI 2023 (arXiv 2307.02848) — SymFormer, SOTA on same benchmark.
- Project page: `mmcheng.net/tb/`, repo `yun-liu/Tuberculosis`.

### Stretch
- SymFormer as drop-in replacement for higher mAP. Same training loop.

### What we're NOT doing
- Training from scratch with our own labels.
- Using a classifier + Grad-CAM pseudo-boxes (defeats the explanation pipeline's grounding).

---

## [2] Dataset

### Training set — TBX11K
- 11,200 CXRs, 512×512, four image classes: `healthy`, `active_tb`, `latent_tb`, `sick_not_tb`. Boxes on TB regions only.
- Splits: 6,600 / 1,800 / 2,800 (canonical; test held out on CodaLab leaderboard).
- Mirrors: Kaggle `vbookshelf/tbx11k-simplified`.
- License: research, non-commercial.

### Auxiliary set — Shenzhen TB annotations (Yang et al. 2022)
- 336 TB-positive Shenzhen cases with **pixel-level** masks for 19 abnormality types (cavity, consolidation, infiltrate, effusion, etc.) + JSON shapes + PNG masks.
- Source: NLM/LHNCBC `data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/`
- Use: train the finding classifier (§3b). Treat each mask as a region label → crop + label → small ResNet18 multi-class head.

### Auxiliary set — PadChest (subset)
- For the fallback view classifier in the quality gate. We only need `Projection` labels (PA / AP / L) — ~160k images labeled. Can get by with 10k-sample subset.

### Calibration / OOD fit
- Use the TBX11K validation split (1,800 images). Fit temperature T (Guo 2017) and Mahalanobis class-conditional mean μ_c, shared covariance Σ on this set.

---

## [3] Quality Gate

### Implementation (hybrid, rule-based + learned components)

```python
def quality_gate(img, dicom_meta) -> QualityStatus:
    # 1. DICOM checks (short-circuit if present)
    if dicom_meta:
        if dicom_meta.ViewPosition not in {"PA", "AP"}:
            return MANUAL_REVIEW  # lateral or unknown
        if dicom_meta.BodyPartExamined and "CHEST" not in dicom_meta.BodyPartExamined.upper():
            return MANUAL_REVIEW

    # 2. View classifier fallback for JPEGs
    if not dicom_meta:
        view = view_classifier(img)  # PadChest-trained ResNet18
        if view == "LATERAL":
            return MANUAL_REVIEW

    # 3. Anatomical sanity via torchxrayvision
    masks = txrv_segment(img)  # 14-class PSPNet, pretrained
    left, right = masks["Left Lung"], masks["Right Lung"]
    if left.sum() < MIN_LUNG_PX or right.sum() < MIN_LUNG_PX:
        return MANUAL_REVIEW  # likely non-chest or severely cropped
    fov_ratio = (left.sum() + right.sum()) / img_size
    if fov_ratio < 0.15:
        return REPEAT_RECOMMENDED
    centroid_offset = abs(lung_centroid_x(left, right) - img_center_x) / img_w
    if centroid_offset > 0.15:
        return BORDERLINE  # rotation/positioning

    # 4. Classical stats
    blur = cv2.Laplacian(img, CV_64F).var()
    if blur < BLUR_THRESH:
        return BORDERLINE
    p1, p99 = np.percentile(img, [1, 99])
    if (p99 - p1) < CONTRAST_MIN:
        return BORDERLINE

    return ACCEPTABLE
```

Thresholds (`MIN_LUNG_PX`, `BLUR_THRESH`, `CONTRAST_MIN`, etc.) tuned on TBX11K train split — held out from detector training for this purpose.

### Dependencies
- `torchxrayvision` — `pip install torchxrayvision`. Ships pretrained PSPNet weights.
- OpenCV for classical stats.
- Tiny ResNet18 (~2M params) fine-tuned on PadChest projection labels.

### Not doing
- Training a single end-to-end quality model. No dataset large enough with the specific quality dimensions we care about.
- Using the Li et al. 2023 technical-adequacy model (weights unreleased).

---

## [4] Deferral

### Three stacked checks

**(a) Temperature scaling.** One learnable scalar T, optimized by minimizing NLL on the validation split. 10 lines of PyTorch:

```python
class TempScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1) * 1.5)
    def forward(self, logits):
        return logits / self.T

# Fit:
ts = TempScale().cuda()
opt = torch.optim.LBFGS([ts.T], lr=0.01, max_iter=50)
def closure():
    opt.zero_grad()
    loss = F.cross_entropy(ts(val_logits), val_labels)
    loss.backward()
    return loss
opt.step(closure)
```

Calibrate detector classification head on validation set; apply T at inference to produce calibrated probability `p_cal = softmax(logits / T)[tb_class]`.

**(b) Selective prediction (Geifman & El-Yaniv 2017).** Two-sided band:

```python
def deferral_decision(p_cal, threshold, margin):
    if p_cal >= threshold + margin:
        return CONFIDENT_POS
    if p_cal <= threshold - margin:
        return CONFIDENT_NEG
    return UNCERTAIN
```

`threshold` is the operating point chosen from the ROC curve (e.g., at WHO 90% sensitivity target). `margin` is chosen from the risk-coverage curve: plot accuracy vs. coverage as margin varies, pick the knee.

**(c) Mahalanobis OOD (Lee 2018 / Anthony & Kamnitsas 2023).**

```python
# Fit (offline, on validation set):
for each class c:
    μ_c = mean(φ(x_i) for x_i in class c)
Σ = shared covariance across classes
Σ_inv = np.linalg.pinv(Σ)

# Inference:
def mahalanobis_score(φ):
    return min((φ - μ_c).T @ Σ_inv @ (φ - μ_c) for c in classes)

if mahalanobis_score(φ(x)) > MAHAL_THRESH:
    return UNCERTAIN  # regardless of p_cal
```

`MAHAL_THRESH` set at the 95th percentile of in-distribution validation Mahalanobis scores. Reference: `HarryAnthony/Mahalanobis-OOD-detection`.

### Final decision

```python
if quality_status in {BORDERLINE, MANUAL_REVIEW, REPEAT_RECOMMENDED}:
    return UNCERTAIN
if mahalanobis_score > thresh:
    return UNCERTAIN
return deferral_decision(p_cal, threshold, margin)
```

### Not doing
- SelectiveNet — requires retraining with a reject head.
- Learning-to-defer (Mozannar & Sontag 2020) — needs paired radiologist labels.
- Monte Carlo dropout / ensembles for uncertainty — too expensive at inference for a prototype.

---

## [5] Explanation module

### Why this architecture (justification for the paper + for ourselves)

Baseline VLM approaches (CXR-LLaVA, LLaVA-Med, CheXagent, MAIRA-2) ingest the image alongside the prompt. Any of them can narrate findings the TB detector never flagged, because they are looking at the image independently. For a clinical-decision-adjacent output, that is unacceptable. The design commitment is: **the LLM does not see the image. It receives a structured record of what the TB detector produced, and its job is prose fluency, not diagnosis.**

Danu et al. 2023 (Siemens, arXiv 2306.10448) — the reference the paper's bib was previously misattributing — use the same decomposition (detector → LLM taking a text prompt with class+probability list) but (a) with a proprietary detector and a proprietary fine-tuned LLM (RadBloomz), (b) with hallucination and repetition reported in their own §5.2. We borrow their two-stage decomposition and swap in: public detector + public-API LLM + grounding-by-construction.

### Stages

**(3a) Anatomical mapping.** Already computed in the quality gate (same torchxrayvision masks reused). Zone assignment:

```python
ANATOMICAL_ZONES = {
    "right_upper": lambda x, y: x > w/2 and y < h/3,
    "right_middle": lambda x, y: x > w/2 and h/3 <= y < 2*h/3,
    "right_lower": lambda x, y: x > w/2 and y >= 2*h/3,
    "left_upper": ...
}
def zone_of(bbox, masks):
    cx, cy = centroid(bbox)
    # verify centroid is inside a lung mask first
    if not (masks["Left Lung"][cy, cx] or masks["Right Lung"][cy, cx]):
        return None  # box not on lung tissue — explanation skipped
    return first_matching_zone(cx, cy)
```

Use 6-zone or 9-zone convention (upper/middle/lower × L/R with or without cardiac/mediastinal zone).

**(3b) Finding classifier.**
- Architecture: ResNet18 multi-class over 19 Shenzhen finding types.
- Input: 224×224 crop from the TBX11K bounding box (+20% padding).
- Training: standard multi-class CE, frozen detector, label = max-intersection Shenzhen mask class.
- Handle class imbalance with weighted sampler.

**(4) Templating.**

```python
TEMPLATE = "{finding} in {side} {zone} lung ({prob:.0%} confidence)"
sentences = [
    TEMPLATE.format(finding=fi, side=si, zone=zi, prob=pi)
    for (fi, si, zi, pi) in detections
]
```

Deterministic. No AI. This is the ground-truth text the LLM must preserve.

**(5) LLM smoothing.** Single API call (Claude Sonnet or GPT-4o):

```
SYSTEM: You are a medical writing assistant. You will receive a structured
list of radiographic findings detected by a TB CAD system. Rewrite the
list as a single fluent clinical paragraph suitable for a radiology
report. Rules:
1. Do not add any finding not in the list.
2. Preserve anatomical location and probabilities exactly.
3. Do not speculate about the cause beyond what the input states.
4. Keep it under 4 sentences.

USER: <the templated sentence list as bullets>
```

No image passed. Token cost per case is trivial (~500 tokens).

**(6) Post-hoc verify.** Parse output with a NER pass (RadGraph model from Jain et al. 2021, or a small regex-based entity matcher over the Shenzhen taxonomy). Reject if any output entity is not in the input set → fall back to bulleted template output.

### Latency budget
- Detector: ~200ms on a single T4 GPU.
- txrv segmentation: ~100ms (reuse from quality gate).
- Finding classifier: ~20ms per box.
- LLM call: 1–3s (network-bound).
- Total: ~2–4s per CXR end-to-end, acceptable for non-urgent screening.

### Not doing
- Fine-tuning CXR-LLaVA / CheXagent / MAIRA-2 — even "light" fine-tuning needs GPU hours we don't have and licensing we don't have.
- Using MAIRA-2 as-is — research-only MSRLA license, and its grounding interface is the wrong direction (it outputs boxes; we need to condition on boxes).
- Training our own VLM — out of scope.

---

## [6] UI

Streamlit for the prototype (fast to build, no separate backend). Single panel:

- Left: original CXR + heatmap overlay + detection boxes
- Right top: TB score, calibrated probability, action routing, AI certainty
- Right middle: findings table (finding, location, confidence)
- Right bottom: natural-language explanation paragraph
- Footer: quality status + any warnings

Export to JSON for logging + React-based dashboard if productionized later.

---

## [7] Monitoring

### Storage
- Event log: append-only Parquet files partitioned by date, plus Postgres for the latest-N queries.
- Per-event fields:
  - `case_id`, `timestamp`, `site_id`, `scanner_model`
  - `quality_status`, `quality_flags[]`
  - `p_cal`, `p_raw`, `mahalanobis_score`, `deferral_decision`
  - `boxes[]` with `(finding, zone, prob)`
  - `feature_vector φ` (downsampled / PCA'd to 64D for storage)
  - patient demographics when available: `age`, `sex`, `hiv_status`

### Metrics computed in nightly batch

```python
# Input-side
daily_volume = count(events, group_by=date)
qc_reject_rate = mean(status != ACCEPTABLE)
qc_breakdown = count(events, group_by=[date, status])
scanner_mix_drift = chi2(current_scanner_mix, training_scanner_mix)

# Prediction-side
score_dist_stats = {mean, std, percentiles} per day
flag_rate = mean(p_cal > threshold)
finding_trajectory = count(boxes, group_by=[date, finding])
brier_on_confirmed = brier_score(p_cal[has_label], y_true[has_label])

# Outcome-side (requires downstream linkage)
override_rate = mean(radiologist_disagreed)
confirmed_tb_yield = sum(bacteriologically_confirmed) / sum(cad_positive)
cascade_completion = sum(reached_treatment) / sum(cad_positive)

# Drift
psi = population_stability_index(score_today, score_reference)
# banded: <0.1 stable, 0.1-0.25 investigate, >0.25 major
ks_stat = scipy.stats.ks_2samp(score_today, score_reference)
mmd = mmd_rbf(φ_today, φ_reference)  # Gretton 2012

# Fairness
for subgroup in [age_band, sex, hiv, site, scanner, qc_status]:
    sens[subgroup] = recall_score(y[subgroup], (p_cal[subgroup] > thresh))
    spec[subgroup] = specificity(y[subgroup], (p_cal[subgroup] > thresh))
    alert_if(subgroup_metric < overall_metric - 2*se)
```

### Alerting
- Slack webhook + email on PSI > 0.25 sustained 7 days, subgroup SE > 2, flag-rate spike > 3σ.
- Dashboard: Streamlit + Plotly for prototype.

### Literature grounding (for paper)
- Finlayson 2021 NEJM: dataset-shift taxonomy (covariate / label / concept).
- Feng 2022 npj Digital Medicine: continual monitoring + silent deployment framework.
- Merkow 2024 Nat Commun: performance alone is an insufficient drift signal; distribution-level tests needed.
- Ghosh 2024 arXiv 2410.13174: CXR-specific scalable drift monitoring.
- Rudolph 2024: empirical post-deployment variation across sites.
- FDA Final PCCP Guidance (Dec 2024): subgroup reporting mandatory.

### Commercial benchmarks (for paper positioning)
- Qure.ai qTrack: cascade tracking aligned with WHO Module 5.
- Delft CAD4TB+ Insights: score histogram, demographics, hotspots, real-time surveillance.
- Lunit INSIGHT Manager 2.0: throughput + real-time image QA.
- Annalise Enterprise: audit logs, prevalence vs. baseline, outlier sites.

---

## Timeline (4 weeks, aggressive)

| Week | Deliverable |
|------|---|
| 1 | TBX11K download, Faster R-CNN fine-tune, basic inference loop. torchxrayvision integrated. Quality gate prototype with hand-tuned thresholds. |
| 2 | Shenzhen classifier trained. Temperature scaling fit. Mahalanobis statistics computed. Deferral decision function end-to-end. |
| 3 | Templating + LLM smoothing + NER verify. Structured evidence UI (Streamlit). Synthetic demo cases. |
| 4 | Monitoring event log schema + batch metrics + dashboard. Paper write-up: results section, example explanation outputs, failure-mode analysis. |

Risk buffer: weeks 1 and 3 are the technical long poles. If Week 1 slips (training infra / dataset access), fall back to inference-only with pretrained weights from the `yun-liu/Tuberculosis` release.

---

## Risks / open questions

1. **TBX11K box labels are region-coarse (`active_tb`, `latent_tb`), not finding-specific.** Shenzhen classifier is the bridge but only has 336 training images. **Plan B:** skip specific finding labels in the explanation; use just location + TB-type ("latent-TB region in right upper lobe, 82% confidence"). Paper should note this as an explicit limitation.

2. **Temperature scaling is population-specific.** Our T from TBX11K validation won't be right for, say, Filipino clinic CXR distribution. Prototype demonstrates the *mechanism*; real deployment requires site-specific recalibration. Note this in the paper's limitations.

3. **Mahalanobis requires per-class μ and shared Σ from training features.** Needs to be extracted during/after detector training, not post-hoc. Minor engineering wrinkle — schedule it into Week 2.

4. **LLM smoothing is a third-party API.** For demo: Claude Sonnet via Anthropic API. For any clinical deployment: self-host Llama-3-8B-Instruct or Qwen2.5-7B. Cost at prototype scale (~100 demo cases): <$5 total. PHI governance is a deployment concern, not a prototype concern.

5. **Post-hoc NER verify may over-reject legitimate paraphrases.** e.g., "cavitation" vs. "cavity." Build entity normalization (small synonym dictionary over the 19-class Shenzhen taxonomy) before blocking.

6. **Outcome-side monitoring metrics (confirmed TB yield, cascade completion) require a real deployment.** For the prototype paper, we describe the metric and pipeline; we don't populate it. Use simulated/retrospective TBX11K-test-set evaluation as stand-in.

7. **Bbox-to-zone assignment can fail** when box centroid falls outside segmented lung (e.g., pleural-based lesion extending into chest wall). Explanation module should skip these with a `location=unspecified` fallback, not crash.

---

## Deliverables map

| Artifact | Location |
|---|---|
| Paper | `tb-cxr-cad/ijacsa-paper.tex` |
| Paper PDF | `tb-cxr-cad/ijacsa-paper.pdf` |
| Outline | `tb-cxr-cad/outline.tex` |
| This PRD | `tb-cxr-cad/implementation-prd.md` |
| Code (not yet) | `tb-cxr-cad/src/` — TBD |
| Configs (not yet) | `tb-cxr-cad/configs/` — TBD |
| Evaluation notebooks (not yet) | `tb-cxr-cad/notebooks/` — TBD |
