# Challenge 4 — Imaging: Architecture & Key Decisions

## Overview

Deploy and evaluate a pre-trained medical imaging AI for chest X-ray (CXR) classification. No fine-tuning required. Judged on demo day with 10 live X-rays across three domains: Performance, Explainability, and Governance.

This repo now supports **two parallel delivery modes**:

1. **Notebook workflow** — Colab / Jupyter-based evaluation and demo flow, used for benchmarking, threshold sweeps, model comparison, and notebook-led presentation.
2. **Web app workflow** — Streamlit-based single-image review app, used for live upload, radiology card generation, GradCAM display, and exportable HTML/PDF outputs.

These are complementary rather than competing approaches: the notebook is the analysis and benchmarking environment, while the web app is the demo-facing user interface.

---

## Evidence Base

| Paper | Authors | Relevance |
|---|---|---|
| CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning | Rajpurkar et al., 2017 (Stanford / Andrew Ng) | Establishes recall ≥ 0.90 as the clinical safety bar for pneumonia detection; validates DenseNet-121 architecture |
| Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization | Selvaraju et al., 2016 | Justifies GradCAM as the XAI method of choice for CNN architectures; human study evidence that it builds appropriate clinical trust |
| Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations | Seyyed-Kalantari et al., 2021 (*Nature Medicine*) | Demonstrates systematic underdiagnosis of female, Black, and Hispanic patients across MIMIC-CXR, CheXpert and NIH — the exact training sources for TorchXRayVision |

---

## Pipeline Flow

```
CXR Image Upload (.jpg / .png)
        │
        ▼
Preprocessing
  ├── Greyscale conversion
  ├── Normalise pixel values to [-1024, 1024] (DICOM range)
  └── Centre-crop + resize to 224×224
        │
        ▼
DenseNet-121 — TorchXRayVision
  ├── Weight set: determined by Step 6b comparison (see Key Decisions)
  ├── Acceptance criterion: pneumonia recall ≥ 0.90 (CheXNet, 2017)
  └── Output: 18 pathology sigmoid scores (multi-label, not mutually exclusive)
        │
        ▼
Threshold Sweep [ADDITION — post Step 6b]
  ├── Plot ROC curve / sensitivity vs threshold
  ├── Find operating threshold achieving ≥ 0.90 pneumonia recall
  └── Replace default 0.30 with evidence-grounded value
        │
        ▼
Output Mapping → 4 Challenge Classes
  ├── Pneumonia        → max(Pneumonia score, Consolidation score)
  ├── COVID-19         → max(Consolidation score, Infiltration score)
  │                       [NOTE: not a labelled training class — declared limitation]
  ├── Pleural Effusion → Effusion score
  └── Normal           → all pathology scores below threshold
        │
        ▼
Clinical Guardrails
  ├── Rule 1: Pneumonia hard flag — any score ≥ 0.20 → EXPEDITED REVIEW
  ├── Rule 2: Uncertainty flag — composite confidence < 0.25 → Senior Radiologist Review
  ├── Rule 3: Normal safety gate — confidence < 0.70 → Indeterminate (Normal withheld)
  ├── Rule 4: Borderline secondary findings — scores 0.20–0.40 → alert flagged
  └── Rule 5 [ADDITION]: Bias declaration
        └── "Normal" verdict is systematically unreliable for under-served populations
            (female, Black, Hispanic, Medicaid patients — Seyyed-Kalantari et al., 2021)
            Human radiologist review required for ALL Normal outputs, not just low-confidence
        │
        ▼
GradCAM — Explainability (Selvaraju et al., 2016)
  ├── Gradients from target pathology class → final DenseNet dense block
  ├── Heatmap overlaid on original X-ray (3-panel: original / heatmap / overlay)
  ├── Radiologist validates anatomical plausibility before acting on output
  └── Expected regions:
        ├── Pneumonia        → lower lobe consolidation / airspace opacity
        ├── Pleural Effusion → costophrenic angle / fluid meniscus
        ├── Consolidation    → lobar or segmental opacity
        └── Normal           → no focal region should strongly dominate
        │
        ▼
GPT-4o-mini — Structured Radiological Report
  ├── Input: 18 pathology probability scores ONLY (no pixel data — GDPR compliance)
  ├── Borderline scores (0.20–0.40) explicitly flagged for clinical attention
  └── Output format: Primary Finding | Secondary Findings | Clinical Recommendation
        │
        ▼
HTML Radiology Card (Demo Day Interface)
  ├── Prediction label + confidence score
  ├── Guardrail alerts (colour-coded by urgency)
  ├── GPT structured report
  ├── Top 6 pathology probability bars
  └── Footer: "For radiologist review only — not for autonomous clinical decision-making"
```

## Delivery Modes

### A. Notebook Workflow

The notebook workflow is the original implementation path described in the challenge spec.

- Runtime: Google Colab / Jupyter
- Primary file: `challenge4_imaging.ipynb`
- Generator source: `generate_notebook.py`
- Purpose:
  - compare TorchXRayVision weight sets
  - run threshold sweeps
  - benchmark on the Step 2 Hugging Face dataset
  - generate plots and evaluation artefacts for slides
  - demonstrate the end-to-end clinical reasoning pipeline in notebook form

**Best use:** model evaluation, evidence gathering, benchmark metrics, appendix figures, and reproducible technical review.

### B. Web App Workflow

The web app workflow is the implemented demo interface for single-image analysis.

- Runtime: Streamlit
- Entry point: `web_app.py`
- Core inference/reporting module: `app/core.py`
- Purpose:
  - upload a local chest X-ray (`.png`, `.jpg`, `.jpeg`, `.dcm`)
  - run DenseNet-121 inference on a single image
  - apply guardrails
  - generate the final radiology card
  - display GradCAM
  - export HTML and PDF summaries

**Best use:** live demo day interaction, rapid case review, presentation flow, and downloadable output artefacts.

## Web App Architecture

### Components

| Component | File(s) | Responsibility |
|---|---|---|
| Streamlit UI | `web_app.py` | Upload flow, demo image picker, result layout, export buttons |
| Inference / reporting engine | `app/core.py` | Preprocessing, DenseNet inference, class mapping, guardrails, GPT report generation, GradCAM, export generation |
| Demo assets | `data/demo_images/` | Local sample images for offline or reliable demo use |
| Model cache | `data/model_cache/` | Cached TorchXRayVision weights stored locally inside the repo |
| Matplotlib cache | `data/mpl_cache/` | Writable plotting cache for restricted Windows environments |
| Startup check | `scripts/startup_check.py` | Confirms model availability and runtime readiness before demo |

### Web App Runtime Flow

```
User Upload / Demo Image Selection
        |
        v
Image Loader
  - PNG / JPG / JPEG via PIL
  - DICOM via pydicom
        |
        v
Preprocessing
  - Greyscale conversion
  - Normalise to [-1024, 1024]
  - Centre-crop + resize to 224x224
        |
        v
TorchXRayVision DenseNet-121
  - Local cached weights
  - 18 pathology sigmoid scores
        |
        +--> Challenge-Class Mapping
        |     - Pneumonia
        |     - COVID-19 pattern proxy
        |     - Pleural Effusion
        |     - Normal
        |
        +--> Clinical Guardrails
        |     - expedited review
        |     - uncertainty escalation
        |     - normal withholding
        |     - bias notice
        |
        +--> Structured Report
        |     - OpenAI GPT report when API key available
        |     - deterministic fallback report otherwise
        |
        +--> GradCAM
        |     - heatmap + overlay
        |
        v
Radiology Card UI
  - result card
  - alerts
  - top findings
  - exports (HTML / PDF)
```

## Data Sources and Assets

### Notebook Path

- Uses the Step 2 Hugging Face chest X-ray dataset for benchmarking and evaluation
- Intended for test-set measurement and slide evidence generation

### Web App Path

- Uses local uploaded images for inference
- Can use local demo images stored in `data/demo_images/`
- Avoids dependence on live web downloads during the demo itself

This separation is intentional: the notebook remains the evaluation environment, while the app is optimized for stable single-image review.

## OpenAI Integration

The two delivery modes use OpenAI differently:

- **Notebook:** GPT-4o-mini is used to turn pathology score vectors into a structured radiology report inside the notebook flow.
- **Web app:** the same score-to-report pattern is used in the app, but with:
  - terminal logging for API call visibility
  - optional prompt logging via environment flag
  - deterministic local fallback reporting if no API key is configured or a request fails

**Privacy model remains unchanged:** pixel data stays local; only numeric pathology scores are sent to the language model.

## Run Requirements

### Notebook

- Colab or Jupyter environment
- notebook dependencies from the generated notebook cells
- OpenAI API key if GPT report generation is required
- internet access for Step 2 dataset and package install steps

### Web App

- local Python environment
- dependencies from `requirements.txt`
- optional `.env` file for API key and runtime settings
- first-run model weight download into `data/model_cache/`
- local browser access to Streamlit

## Demo Day Recommendation

Run both tracks in parallel:

- **Notebook presenter:** owns benchmarking, threshold evidence, model comparison, and research-backed evaluation narrative
- **Web app presenter:** owns live image upload, radiology card demo, GradCAM walkthrough, and exported output artefacts

This gives the team both:

- a rigorous evaluation story
- a polished user-facing demonstration

---

## Key Decisions

### 1. Weight Set Selection
Run Step 6b to compare all 5 weight sets on the test set:

| Weights | Training Data | Notes |
|---|---|---|
| `densenet121-res224-all` | All datasets combined | Best generalisation to unknown distribution |
| `densenet121-res224-rsna` | RSNA Pneumonia Challenge | Highest pneumonia sensitivity candidate |
| `densenet121-res224-nih` | NIH ChestX-ray14 | Closest to CheXNet's original training setup |
| `densenet121-res224-chex` | CheXpert (Stanford) | High-quality PA film performance |
| `densenet121-res224-mimic_nb` | MIMIC-CXR (MIT/BIDMC) | Large US hospital population |

**Decision criterion:** Select the weight set that achieves pneumonia recall ≥ 0.90 at the lowest threshold (i.e. without sacrificing specificity unnecessarily). If multiple sets meet this bar, prefer `all` for generalisation.

### 2. Operating Threshold
- Run a threshold sweep (0.10 → 0.50 in 0.05 steps) after Step 6b
- Plot pneumonia sensitivity and specificity at each threshold
- Set final threshold at the point where pneumonia recall first reaches ≥ 0.90
- Document the tradeoff: lower threshold = higher sensitivity, more false positives
- Cite CheXNet as justification for the ≥ 0.90 target

**Insert after Step 6b in the notebook:**

```python
# Threshold Sweep — find operating point achieving pneumonia recall >= 0.90
# Evidence base: Rajpurkar et al. (2017) CheXNet — radiologist-level detection requires recall >= 0.90

from sklearn.metrics import recall_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Re-use all_probs and y_true from Step 5
# all_probs: list of 18-d numpy arrays (one per image)
# y_true:    list of string labels e.g. 'PNEUMONIA' / 'NORMAL'

THRESHOLDS = np.arange(0.10, 0.55, 0.05)

sweep_results = []

for thresh in THRESHOLDS:
    y_pred_sweep = []
    for probs in all_probs:
        pneumonia_score = max(probs[PNEUMONIA_IDX], probs[CONSOLID_IDX])
        pred = 'PNEUMONIA' if pneumonia_score >= thresh else 'NORMAL'
        y_pred_sweep.append(pred)

    y_true_bin = [1 if y == 'PNEUMONIA' else 0 for y in y_true]
    y_pred_bin = [1 if y == 'PNEUMONIA' else 0 for y in y_pred_sweep]

    recall    = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    acc       = accuracy_score(y_true_bin, y_pred_bin)

    sweep_results.append({
        'threshold': round(thresh, 2),
        'recall':    round(recall, 3),
        'precision': round(precision, 3),
        'accuracy':  round(acc, 3),
    })

# Print summary table
print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} {'Accuracy':>10}")
print('-' * 42)
for r in sweep_results:
    marker = ' <-- target' if r['recall'] >= 0.90 else ''
    print(f"{r['threshold']:>10.2f} {r['recall']:>8.3f} {r['precision']:>10.3f} {r['accuracy']:>10.3f}{marker}")

# Find the highest threshold that still achieves recall >= 0.90
# (highest threshold = fewest false positives while meeting the safety bar)
candidates = [r for r in sweep_results if r['recall'] >= 0.90]
if candidates:
    chosen = max(candidates, key=lambda r: r['threshold'])
    print(f"\nChosen threshold: {chosen['threshold']}")
    print(f"  Pneumonia recall:    {chosen['recall']*100:.1f}%  (target >= 90%)")
    print(f"  Precision:           {chosen['precision']*100:.1f}%")
    print(f"  Accuracy:            {chosen['accuracy']*100:.1f}%")
    THRESHOLD = chosen['threshold']  # update global threshold
else:
    print("\nWARNING: No threshold achieves >= 0.90 recall on this test set.")
    print("Consider switching weight set (see Step 6b) or lowering threshold to 0.10.")
    THRESHOLD = 0.10  # fallback — maximise safety

# Plot sensitivity vs threshold
recalls    = [r['recall']    for r in sweep_results]
precisions = [r['precision'] for r in sweep_results]
thresholds = [r['threshold'] for r in sweep_results]

plt.figure(figsize=(8, 5))
plt.plot(thresholds, recalls,    marker='o', label='Pneumonia Sensitivity (Recall)')
plt.plot(thresholds, precisions, marker='s', label='Precision', linestyle='--')
plt.axhline(0.90, color='red', linestyle=':', linewidth=1.5, label='CheXNet safety bar (0.90)')
if candidates:
    plt.axvline(chosen['threshold'], color='green', linestyle=':', linewidth=1.5,
                label=f"Chosen threshold ({chosen['threshold']})")
plt.xlabel('Classification Threshold')
plt.ylabel('Score')
plt.title('Threshold Sweep — Pneumonia Sensitivity vs Precision\n(Rajpurkar et al. 2017: recall ≥ 0.90 required)')
plt.legend()
plt.tight_layout()
plt.savefig('threshold_sweep.png', dpi=120)
plt.show()
print('Saved: threshold_sweep.png')
```

### 3. COVID-19 Handling
- COVID-19 was not a labelled class in any training dataset (pre-pandemic data)
- Model detects radiological features (bilateral consolidation, ground-glass opacity) but not the aetiology
- This must be declared explicitly in the model card and on Slide 1
- Do not present COVID-19 predictions as reliable — flag as "radiological features consistent with COVID-19; aetiology unconfirmed"

### 4. Normal Verdict Policy
- Current guardrail: withhold Normal if confidence < 0.70
- Additional policy (Seyyed-Kalantari et al., 2021): Normal is unreliable for under-served groups regardless of confidence score
- All Normal outputs require human radiologist review — this is non-negotiable and must be stated on the radiology card and governance slide

---

## Scoring Grid

| Domain | Criterion | Marks |
|---|---|---|
| Performance (Slide 1) | Classification accuracy — 10 live images × 1pt (missed pneumonia = 0) | 10 |
| Performance (Slide 1) | GPT radiology report quality — 5 judge images × 1pt | 5 |
| Explainability (Slide 2) | Model card: data provenance, known limitations, XAI method | 5 |
| Governance (Slide 3) | Acceptable use, data privacy, monitoring, compliance, liability | 5 |
| **Total** | | **25** |

---

## Slide Structure

### Slide 1 — Performance
- Which weight set was selected and why (benchmark against all 5)
- Operating threshold chosen and evidence base (CheXNet recall ≥ 0.90)
- Accuracy, pneumonia sensitivity, confusion matrix on 200-image test set
- CheXNet comparison: "radiologist-level performance claimed — tempered by Seyyed-Kalantari et al. (2021) bias evidence"

### Slide 2 — Explainability (Model Card)
- Architecture: DenseNet-121, ~7M parameters, 224×224 greyscale input, 18-output sigmoid
- Training data provenance: NIH (112k), CheXpert (224k), MIMIC-CXR (370k), PadChest (160k), RSNA (30k)
- XAI method: GradCAM (Selvaraju et al., 2016) — show heatmap from ≥1 pneumonia image
- Known limitations:
  - COVID-19 not a labelled training class
  - Systematic underdiagnosis of under-served populations (Seyyed-Kalantari et al., 2021)
  - Validated on PA films only; not for mobile/AP X-rays
  - GPT-4o-mini may generate plausible but inaccurate clinical statements
  - Threshold not clinically validated; selected empirically on test set

### Slide 3 — Governance
- **Acceptable use:** Radiologist triage support, medical education, research only. Not for autonomous reporting, paediatric patients, or emergency settings without human confirmation
- **Data privacy:** Pixel data never leaves Colab; GPT-4o-mini receives probability scores only; NHS DSPT + GDPR Article 9 compliance required for deployment
- **Monitoring:** Monthly radiologist audit of 20 random predictions; **stratified by sex and ethnicity** (Seyyed-Kalantari et al., 2021); alert if pneumonia sensitivity drops below 0.80; quarterly re-evaluation against fixed hold-out set
- **Compliance:** MHRA SaMD Class IIa/IIb; NHS AI Framework; RCR AI guidance; UK GDPR Article 22
- **Liability:** Clinician remains decision-maker under current UK law; AI output is advisory only; NHS trust must verify indemnity covers AI-assisted decisions

---

## Known Limitations (Model Card — full list)

1. COVID-19 is not a labelled training class; model detects radiological features only
2. Systematic underdiagnosis of female, Black, Hispanic, and Medicaid patients — all Normal outputs require human review (Seyyed-Kalantari et al., 2021)
3. Trained on US and European hospital populations; may underperform on other populations
4. Validated on PA (posteroanterior) films only; not for AP or mobile X-rays
5. GPT-4o-mini reports may generate plausible but clinically inaccurate statements
6. Operating threshold selected empirically on test set; not externally validated
7. GradCAM heatmaps require radiologist interpretation — model attention is not a diagnosis
