"""
Challenge 4 — Imaging: Notebook Generator
Run locally: python generate_notebook.py
Produces:    challenge4_imaging.ipynb  → upload to Colab and run top-to-bottom.
"""

import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}


# ── Cells ─────────────────────────────────────────────────────────────────────

CELLS = []

# ── Title ─────────────────────────────────────────────────────────────────────
CELLS.append(md("""\
# Challenge 4 — Imaging: Chest X-Ray Classification
### Oxford Clinical AI Hackathon

**Mission:** Deploy and evaluate a pre-trained medical imaging AI for chest X-ray classification.
No fine-tuning required. Judged on demo day with 10 live X-rays.

**Evidence base:**
- Rajpurkar et al. 2017 — CheXNet: recall ≥ 0.90 = clinical safety bar for pneumonia
- Selvaraju et al. 2016 — GradCAM: XAI method for CNN architectures
- Seyyed-Kalantari et al. 2021 (*Nature Medicine*) — underdiagnosis bias in under-served populations

---
"""))

# ── Setup ─────────────────────────────────────────────────────────────────────
CELLS.append(md("## Setup — API Key and OpenAI Client"))

CELLS.append(code("""\
from google.colab import userdata
import os, json
from openai import OpenAI

client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))
print('OK OpenAI client ready.')
"""))

# ── Step 1: Install ───────────────────────────────────────────────────────────
CELLS.append(md("## Step 1 — Install Libraries"))

CELLS.append(code("""\
!pip install -q torchxrayvision scikit-image datasets Pillow scikit-learn python-pptx

import torch
import torchxrayvision as xrv
import skimage
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from collections import Counter

print('OK All libraries loaded.')
print(f'TorchXRayVision version: {xrv.__version__}')
print(f'PyTorch: {torch.__version__}')
"""))

# ── Step 2: Dataset ───────────────────────────────────────────────────────────
CELLS.append(md("""\
## Step 2 — Load the Test Dataset

We use the HuggingFace chest X-ray dataset (test split only — nothing to train).
This is **out-of-distribution evaluation**: data the model has never seen, from a
different source than its training data. This tells you far more about real-world
clinical performance than validation accuracy.
"""))

CELLS.append(code("""\
import random
from datasets import load_dataset
from collections import Counter

print('Loading chest X-ray test set from HuggingFace...')
dataset = load_dataset('KVG-DataScience/chest_xray_dataset')

test_ds_full = dataset['test']

NUM_SAMPLES = 200
if len(test_ds_full) > NUM_SAMPLES:
    test_ds = test_ds_full.shuffle(seed=42).select(range(NUM_SAMPLES))
else:
    test_ds = test_ds_full

LABELS = test_ds.features['label'].names
print(f'Test images: {len(test_ds)}')
print(f'Classes: {LABELS}')

counts = Counter(test_ds['label'])
print('\\nTest set distribution:')
for i, l in enumerate(LABELS):
    print(f'  {l:20s}: {counts.get(i, 0):>4} images')
"""))

# ── Step 3: Load model ────────────────────────────────────────────────────────
CELLS.append(md("""\
## Step 3 — Load TorchXRayVision Pre-Trained Model

We start with `densenet121-res224-all` (trained on all datasets) — the most robust
choice for an unknown test set. We compare weight sets in Step 6b.

| Weights | Training Data | Notes |
|---|---|---|
| `densenet121-res224-all` | All datasets combined | Best generalisation |
| `densenet121-res224-rsna` | RSNA Pneumonia Challenge | Highest pneumonia sensitivity |
| `densenet121-res224-nih` | NIH ChestX-ray14 | Closest to CheXNet setup |
| `densenet121-res224-chex` | CheXpert (Stanford) | High-quality PA films |
| `densenet121-res224-mimic_nb` | MIMIC-CXR (MIT/BIDMC) | Large US hospital population |
"""))

CELLS.append(code("""\
WEIGHTS = 'densenet121-res224-all'  # updated in Step 6b after comparison

print(f'Loading TorchXRayVision: {WEIGHTS}...')
model = xrv.models.DenseNet(weights=WEIGHTS)
model.eval()

print(f'OK Model loaded.')
print(f'Architecture: DenseNet-121')
print(f'Input size:   224x224 (greyscale)')
print(f'Output:       {len(model.pathologies)} pathologies (sigmoid)')
print(f'\\nAll pathologies the model can detect:')
for i, p in enumerate(model.pathologies):
    print(f'  {i+1:2d}. {p}')
"""))

# ── Step 4: Mapping ───────────────────────────────────────────────────────────
CELLS.append(md("""\
## Step 4 — Map Model Outputs to Challenge Classes

The model outputs 18 pathology scores. Our challenge has 4 classes.

| Challenge Class | TorchXRayVision Output(s) |
|---|---|
| Pneumonia | Pneumonia + Consolidation |
| COVID-19 | Consolidation + Infiltration *(not a labelled training class — see limitation)* |
| Pleural Effusion | Effusion |
| Normal | All pathology scores below threshold |

> **COVID-19 limitation:** TorchXRayVision was trained on pre-pandemic data.
> COVID-19 as a labelled class does not exist. The model may detect radiological
> features (bilateral consolidation, ground-glass opacity) without knowing the
> aetiology. This must be declared in the model card.
"""))

CELLS.append(code("""\
def get_pathology_idx(name):
    return model.pathologies.index(name) if name in model.pathologies else None

PNEUMONIA_IDX = get_pathology_idx('Pneumonia')
CONSOLID_IDX  = get_pathology_idx('Consolidation')
INFILTR_IDX   = get_pathology_idx('Infiltration')
EFFUSION_IDX  = get_pathology_idx('Effusion')

print('Pathology index mapping:')
print(f'  Pneumonia      (index {PNEUMONIA_IDX})')
print(f'  Consolidation  (index {CONSOLID_IDX})')
print(f'  Infiltration   (index {INFILTR_IDX})')
print(f'  Effusion       (index {EFFUSION_IDX})')

# Default threshold — overwritten by threshold sweep in Step 6c
THRESHOLD = 0.30

def predict_challenge_class(probs, threshold=THRESHOLD):
    \"\"\"Map 18-pathology sigmoid outputs to challenge class.\"\"\"
    pneumonia_score = max(probs[PNEUMONIA_IDX], probs[CONSOLID_IDX])
    if pneumonia_score >= threshold:
        return 'PNEUMONIA', pneumonia_score, {'PNEUMONIA': pneumonia_score}
    else:
        return 'NORMAL', 1 - pneumonia_score, {'PNEUMONIA': pneumonia_score}

print(f'\\nOK predict_challenge_class() ready.')
print(f'Detection threshold: {THRESHOLD}  (will be tuned in Step 6c)')
"""))

# ── Step 5: Preprocessing + inference ────────────────────────────────────────
CELLS.append(md("""\
## Step 5 — Preprocess and Run Inference on Test Set

TorchXRayVision preprocessing pipeline:
- Images must be greyscale (single channel)
- Pixel values normalised to `[-1024, 1024]` (DICOM range)
- Centre-cropped and resized to 224×224
"""))

CELLS.append(code("""\
transform = T.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224),
])

def preprocess_image(pil_img):
    \"\"\"Convert PIL image to TorchXRayVision tensor.\"\"\"
    img = np.array(pil_img.convert('L'))          # greyscale
    img = xrv.datasets.normalize(img, 255)         # [0,255] -> [-1024,1024]
    img = img[None, ...]                            # add channel dim
    img = transform(img)                            # crop + resize
    return torch.from_numpy(img).unsqueeze(0)       # add batch dim

# Run inference on entire test set
print('Running inference on test set...')
y_true, y_pred, all_probs = [], [], []

for i, sample in enumerate(test_ds):
    img_tensor = preprocess_image(sample['image'])
    with torch.no_grad():
        probs = torch.sigmoid(model(img_tensor))[0].numpy()
    pred_class, score, _ = predict_challenge_class(probs)
    y_true.append(LABELS[sample['label']])
    y_pred.append(pred_class)
    all_probs.append(probs)
    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{len(test_ds)} done...')

print(f'\\nOK Inference complete on {len(test_ds)} images.')
"""))

# ── Step 6: Evaluate ──────────────────────────────────────────────────────────
CELLS.append(md("""\
## Step 6 — Evaluate: Accuracy, Sensitivity, Confusion Matrix

The judges will score your model on a 10-image live test set at demo time.
This full evaluation tells you what to expect.

**Pneumonia sensitivity is the safety-critical metric** — a missed pneumonia = 0 pts for that image.
Evidence base: Rajpurkar et al. 2017 (CheXNet) — recall ≥ 0.90 required for radiologist-level performance.
"""))

CELLS.append(code("""\
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, recall_score
)

acc = accuracy_score(y_true, y_pred)
print(f'Overall Accuracy: {acc*100:.1f}%')

print('\\n--- Per-Class Report ---')
print(classification_report(y_true, y_pred, labels=LABELS, digits=3))

pneumonia_recall = recall_score(
    [1 if y == 'PNEUMONIA' else 0 for y in y_true],
    [1 if y == 'PNEUMONIA' else 0 for y in y_pred]
)
print(f'Pneumonia sensitivity: {pneumonia_recall*100:.1f}%')
print(f'CheXNet safety bar:    90.0%  (Rajpurkar et al. 2017)')
if pneumonia_recall < 0.90:
    print('WARNING: Below safety bar — adjust threshold in Step 6c.')

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred, labels=LABELS, ax=ax, colorbar=False
)
ax.set_title(f'Confusion Matrix — TorchXRayVision ({WEIGHTS})', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120)
plt.show()
print('Saved: confusion_matrix.png')
"""))

# ── Step 6b: Weight set comparison ───────────────────────────────────────────
CELLS.append(md("""\
## Step 6b — Model Comparison: Which Weight Set Performs Best?

Rather than training a model, we select the best pre-trained model for this
specific clinical task. This is a genuine real-world clinical AI skill.

**Decision criterion (from architecture):** Select the weight set that achieves
pneumonia recall ≥ 0.90 at the lowest threshold. If multiple sets meet this bar,
prefer `all` for generalisation.
"""))

CELLS.append(code("""\
from sklearn.metrics import accuracy_score, recall_score

WEIGHT_SETS = [
    'densenet121-res224-all',
    'densenet121-res224-rsna',
    'densenet121-res224-nih',
    'densenet121-res224-chex',
    'densenet121-res224-mimic_nb',
]

comparison_results = {}

for w in WEIGHT_SETS:
    print(f'Testing {w}...')
    try:
        m = xrv.models.DenseNet(weights=w)
        m.eval()
        yt, yp, wt_probs = [], [], []

        for sample in test_ds:
            img_tensor = preprocess_image(sample['image'])
            with torch.no_grad():
                probs = torch.sigmoid(m(img_tensor))[0].numpy()

            p_idx = m.pathologies.index('Pneumonia')    if 'Pneumonia'     in m.pathologies else None
            c_idx = m.pathologies.index('Consolidation') if 'Consolidation' in m.pathologies else None

            if p_idx is not None and c_idx is not None:
                score = max(probs[p_idx], probs[c_idx])
            elif p_idx is not None:
                score = probs[p_idx]
            else:
                score = 0.0

            pred = 'PNEUMONIA' if score >= THRESHOLD else 'NORMAL'
            yt.append(LABELS[sample['label']])
            yp.append(pred)
            wt_probs.append(probs)

        acc_w = accuracy_score(yt, yp)
        rec_w = recall_score(
            [1 if y == 'PNEUMONIA' else 0 for y in yt],
            [1 if y == 'PNEUMONIA' else 0 for y in yp]
        )
        comparison_results[w] = {
            'accuracy': acc_w,
            'pneumonia_recall': rec_w,
            'probs': wt_probs,
            'y_pred': yp,
        }
        meets_bar = 'YES' if rec_w >= 0.90 else 'NO '
        print(f'  Accuracy: {acc_w*100:.1f}% | Pneumonia recall: {rec_w*100:.1f}%  | Meets CheXNet bar: {meets_bar}')

    except Exception as e:
        print(f'  Skipped: {e}')
        comparison_results[w] = None

print('\\n--- Summary ---')
print(f'{\"Weights\":<35} {\"Accuracy\":>10} {\"Pneu. Recall\":>14} {\"Meets bar\":>10}')
print('-' * 72)
for w, r in comparison_results.items():
    if r:
        bar = 'YES' if r['pneumonia_recall'] >= 0.90 else 'NO'
        print(f'{w:<35} {r[\"accuracy\"]*100:>9.1f}% {r[\"pneumonia_recall\"]*100:>13.1f}%  {bar:>10}')

# Select best weight set
candidates = {w: r for w, r in comparison_results.items()
              if r and r['pneumonia_recall'] >= 0.90}
if candidates:
    # prefer highest threshold (most conservative) that meets bar
    best = max(candidates, key=lambda w: candidates[w]['accuracy'])
    print(f'\\nSelected weight set: {best}')
    WEIGHTS = best
    model = xrv.models.DenseNet(weights=WEIGHTS)
    model.eval()
    # Rebuild PNEUMONIA_IDX etc for new model
    PNEUMONIA_IDX = get_pathology_idx('Pneumonia')
    CONSOLID_IDX  = get_pathology_idx('Consolidation')
    INFILTR_IDX   = get_pathology_idx('Infiltration')
    EFFUSION_IDX  = get_pathology_idx('Effusion')
    # Re-run inference with best model
    y_true, y_pred, all_probs = [], [], []
    for i, sample in enumerate(test_ds):
        img_tensor = preprocess_image(sample['image'])
        with torch.no_grad():
            probs = torch.sigmoid(model(img_tensor))[0].numpy()
        pred_class, score, _ = predict_challenge_class(probs)
        y_true.append(LABELS[sample['label']])
        y_pred.append(pred_class)
        all_probs.append(probs)
    print(f'Re-ran inference with {WEIGHTS}. Proceed to Step 6c.')
else:
    print('\\nWARNING: No weight set achieved recall >= 0.90 at current threshold.')
    print('Proceeding to threshold sweep (Step 6c) — lower threshold may fix this.')
"""))

# ── Step 6c: Threshold sweep (our addition) ───────────────────────────────────
CELLS.append(md("""\
## Step 6c — Threshold Sweep: Evidence-Grounded Operating Point

**This step is an addition to the original notebook.**

CheXNet (Rajpurkar et al. 2017) established that radiologist-level pneumonia
detection requires recall ≥ 0.90. We sweep thresholds to find the highest
threshold that still meets this safety bar — minimising false positives while
not missing pneumonia cases.
"""))

CELLS.append(code("""\
from sklearn.metrics import recall_score, precision_score, accuracy_score
import numpy as np

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

print(f'Threshold sweep results (weight set: {WEIGHTS})')
print(f'CheXNet safety bar: recall >= 0.90  (Rajpurkar et al. 2017)')
print()
print(f'{\"Threshold\":>10} {\"Recall\":>8} {\"Precision\":>10} {\"Accuracy\":>10}')
print('-' * 44)
for r in sweep_results:
    marker = '  <-- meets safety bar' if r['recall'] >= 0.90 else ''
    print(f'{r[\"threshold\"]:>10.2f} {r[\"recall\"]:>8.3f} {r[\"precision\"]:>10.3f} {r[\"accuracy\"]:>10.3f}{marker}')

# Choose highest threshold that still achieves recall >= 0.90
candidates = [r for r in sweep_results if r['recall'] >= 0.90]
if candidates:
    chosen = max(candidates, key=lambda r: r['threshold'])
    print(f'\\nChosen threshold: {chosen[\"threshold\"]}')
    print(f'  Pneumonia recall:    {chosen[\"recall\"]*100:.1f}%  (target >= 90%)')
    print(f'  Precision:           {chosen[\"precision\"]*100:.1f}%')
    print(f'  Accuracy:            {chosen[\"accuracy\"]*100:.1f}%')
    THRESHOLD = chosen['threshold']
else:
    print('\\nWARNING: No threshold achieves >= 0.90 recall. Using 0.10 (maximum sensitivity).')
    THRESHOLD = 0.10

# Plot
recalls    = [r['recall']    for r in sweep_results]
precisions = [r['precision'] for r in sweep_results]
thresholds = [r['threshold'] for r in sweep_results]

plt.figure(figsize=(8, 5))
plt.plot(thresholds, recalls,    marker='o', label='Pneumonia Sensitivity (Recall)')
plt.plot(thresholds, precisions, marker='s', label='Precision', linestyle='--')
plt.axhline(0.90, color='red', linestyle=':', linewidth=1.5, label='CheXNet safety bar (0.90)')
if candidates:
    plt.axvline(chosen['threshold'], color='green', linestyle=':',
                linewidth=1.5, label=f'Chosen threshold ({chosen[\"threshold\"]})')
plt.xlabel('Classification Threshold')
plt.ylabel('Score')
plt.title(f'Threshold Sweep — {WEIGHTS}\\nCheXNet safety bar: recall >= 0.90 (Rajpurkar et al. 2017)')
plt.legend()
plt.tight_layout()
plt.savefig('threshold_sweep.png', dpi=120)
plt.show()
print('Saved: threshold_sweep.png')
print(f'\\nFinal THRESHOLD set to: {THRESHOLD}')
"""))

# ── Step 7: GPT reports ───────────────────────────────────────────────────────
CELLS.append(md("""\
## Step 7 — Generate Radiological Reports with GPT-4o-mini

The DenseNet classifies across 18 pathologies. GPT-4o-mini articulates those
findings as a structured consultant-radiologist report.

We pass the **full 18-pathology probability vector** — not just the top prediction.
This gives the language model richer clinical context and allows it to note
secondary findings.

> **Privacy note:** GPT-4o-mini receives probability scores only — no pixel data,
> no patient identifiers. This is GDPR Article 9 compliant.
"""))

CELLS.append(code("""\
RADIOLOGY_SYS = (
    'You are a consultant radiologist generating a structured chest X-ray report. '
    'You will receive a vector of probability scores (0-1) for 18 radiological findings '
    'from a pre-trained DenseNet-121 model (TorchXRayVision). '
    'Use this exact format: '
    'CHEST X-RAY REPORT | PRIMARY FINDING | SECONDARY FINDINGS | CLINICAL RECOMMENDATION. '
    'Note any borderline scores (0.20-0.40) as requiring clinical attention. '
    'Do not exceed 5 sentences. Be clinically precise.'
)

def classify_and_report(image_path_or_pil, title='Chest X-Ray'):
    \"\"\"Full pipeline: preprocess -> DenseNet inference -> GPT-4o-mini report.\"\"\"
    if isinstance(image_path_or_pil, str):
        pil_img = Image.open(image_path_or_pil)
    else:
        pil_img = image_path_or_pil

    img_tensor = preprocess_image(pil_img)
    with torch.no_grad():
        probs = torch.sigmoid(model(img_tensor))[0].numpy()

    full_findings = {p: round(float(v), 3)
                     for p, v in zip(model.pathologies, probs)}

    pred_class, confidence, class_scores = predict_challenge_class(probs, THRESHOLD)

    prompt = (
        f'18-pathology probability scores from DenseNet-121: {full_findings}. '
        f'Top challenge-class prediction: {pred_class} '
        f'(composite confidence {confidence:.3f}).\\n'
        'Generate a structured radiological report.'
    )

    r = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': RADIOLOGY_SYS},
            {'role': 'user',   'content': prompt}
        ],
        temperature=0.1,
        max_tokens=300
    )
    report = r.choices[0].message.content
    return pred_class, confidence, full_findings, report

print('OK classify_and_report() ready.')
"""))

# ── Step 8: GradCAM ───────────────────────────────────────────────────────────
CELLS.append(md("""\
## Step 8 — XAI: Explain the Model with GradCAM

**Method:** GradCAM — Gradient-weighted Class Activation Mapping (Selvaraju et al. 2016).

Computes gradients flowing back from the predicted class score into the final
convolutional layer, then visualises which spatial regions activated most strongly.

**Why this matters clinically:** A model that correctly classifies "Pneumonia" but
attends to the left shoulder instead of the right lower lobe is not clinically
trustworthy — it has learned a spurious correlation. GradCAM lets a radiologist
verify the model is looking in the right place.

Expected regions:
- **Pneumonia** → lower lobe consolidation / airspace opacity
- **Effusion** → costophrenic angle / fluid meniscus
- **Normal** → no focal region should strongly dominate

> Reference: Selvaraju et al. 2016 — works on any CNN without retraining;
> human study evidence that it builds appropriate clinical trust.
"""))

CELLS.append(code("""\
import torch.nn.functional as F

def gradcam(model, img_tensor, target_class_name):
    \"\"\"Compute GradCAM heatmap for a target pathology class.\"\"\"
    if target_class_name not in model.pathologies:
        print(f'Warning: {target_class_name} not in model pathologies. Using Pneumonia.')
        target_class_name = 'Pneumonia'

    target_idx = model.pathologies.index(target_class_name)
    gradients  = []
    activations= []

    def save_gradient(grad):
        gradients.append(grad)

    def save_activation(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    handle = model.features.denseblock4.register_forward_hook(save_activation)

    output = torch.sigmoid(model(img_tensor))
    handle.remove()

    model.zero_grad()
    output[0, target_idx].backward(retain_graph=True)

    pooled_grads = gradients[0].mean(dim=[0, 2, 3])
    act = activations[0][0]
    for i in range(act.shape[0]):
        act[i, :, :] *= pooled_grads[i]

    heatmap = act.mean(dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def display_gradcam(image_path_or_pil, target='Pneumonia'):
    \"\"\"Classify, compute GradCAM, and display a 3-panel visualisation.\"\"\"
    if isinstance(image_path_or_pil, str):
        pil_img = Image.open(image_path_or_pil).convert('L')
    else:
        pil_img = image_path_or_pil.convert('L')

    img_tensor = preprocess_image(pil_img)
    heatmap = gradcam(model, img_tensor, target)

    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_img.size, Image.LANCZOS)
    ) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    orig = np.array(pil_img)

    axes[0].imshow(orig, cmap='gray')
    axes[0].set_title('Original X-Ray')
    axes[0].axis('off')

    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title(f'GradCAM — {target}')
    axes[1].axis('off')

    axes[2].imshow(orig, cmap='gray')
    axes[2].imshow(heatmap_resized, cmap='jet', alpha=0.45)
    axes[2].set_title('Overlay — What the model attended to')
    axes[2].axis('off')

    plt.suptitle(
        f'XAI: GradCAM for {target} — DenseNet-121 ({WEIGHTS})\\n'
        'Selvaraju et al. 2016 — gradient-weighted class activation mapping',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('gradcam_output.png', dpi=120)
    plt.show()
    print('Saved: gradcam_output.png')
    print()
    print('Radiologist should verify heatmap attends to:')
    print('  Pneumonia        -> lower lobe consolidation / airspace opacity')
    print('  Effusion         -> costophrenic angle / fluid meniscus')
    print('  Consolidation    -> lobar or segmental opacity')
    print('  Normal           -> no focal region should strongly dominate')


print('OK display_gradcam() ready.')
print('Usage: display_gradcam(pil_image, target="Pneumonia")')
"""))

# ── Step 9: Clinical guardrails ───────────────────────────────────────────────
CELLS.append(md("""\
## Step 9 — Clinical Guardrails

The model outputs continuous probability scores — not hard binary decisions.
A safe clinical deployment must define explicit rules for when to escalate,
when to flag uncertainty, and when to refuse a \"Normal\" verdict.

**Evidence base:**
- Rajpurkar et al. 2017 — CheXNet: recall ≥ 0.90 required for radiologist-level pneumonia detection
- NHS England AI imaging guidance: human-in-the-loop required for all AI-assisted reporting
- Seyyed-Kalantari et al. 2021 — \"Normal\" systematically unreliable for under-served populations:
  female, Black, Hispanic, Medicaid patients. All Normal outputs require human review.
"""))

CELLS.append(code("""\
PNEUMONIA_HARD_FLAG_THRESHOLD = 0.20   # any score above this -> mandatory expedited review
UNCERTAINTY_THRESHOLD         = 0.25   # composite confidence below this -> senior radiologist
NORMAL_SAFETY_THRESHOLD       = 0.70   # Normal only returned if confidence above this

def apply_guardrails(pred_class, confidence, full_findings):
    \"\"\"Apply clinical safety rules to model output.\"\"\"
    alerts = []

    # Rule 1: Pneumonia hard flag
    pneu_raw = max(
        full_findings.get('Pneumonia', 0),
        full_findings.get('Consolidation', 0)
    )
    if pneu_raw >= PNEUMONIA_HARD_FLAG_THRESHOLD:
        alerts.append(
            f'EXPEDITED REVIEW — Pneumonia/Consolidation score {pneu_raw:.2f} '
            f'>= {PNEUMONIA_HARD_FLAG_THRESHOLD} (CheXNet threshold)'
        )

    # Rule 2: Uncertainty flag
    if confidence < UNCERTAINTY_THRESHOLD:
        alerts.append(
            f'UNCERTAIN — Max composite score {confidence:.2f} '
            f'< {UNCERTAINTY_THRESHOLD} — Senior Radiologist review required'
        )

    # Rule 3: Normal safety gate
    if pred_class == 'NORMAL' and confidence < NORMAL_SAFETY_THRESHOLD:
        pred_class = 'Indeterminate'
        alerts.append(
            f'Normal withheld — confidence {confidence:.2f} '
            f'< {NORMAL_SAFETY_THRESHOLD} safety threshold'
        )

    # Rule 4: Borderline secondary findings
    borderline = [(p, v) for p, v in full_findings.items()
                  if 0.20 <= v <= 0.40 and p not in ['Pneumonia', 'Consolidation']]
    if borderline:
        bl_str = ', '.join(f'{p} ({v:.2f})' for p, v in sorted(borderline, key=lambda x: -x[1]))
        alerts.append(f'Borderline secondary findings: {bl_str}')

    # Rule 5: Bias declaration (Seyyed-Kalantari et al. 2021)
    # "Normal" is systematically unreliable for under-served populations.
    # All Normal outputs require human review regardless of confidence.
    if pred_class in ('NORMAL', 'Normal'):
        alerts.append(
            'BIAS NOTICE — "Normal" verdict unreliable for under-served populations '
            '(female, Black, Hispanic, Medicaid patients). '
            'Human radiologist review required. (Seyyed-Kalantari et al. 2021)'
        )

    return pred_class, alerts


print('OK Clinical guardrails defined.')
print(f'Pneumonia flag threshold:  >= {PNEUMONIA_HARD_FLAG_THRESHOLD}')
print(f'Uncertainty threshold:      < {UNCERTAINTY_THRESHOLD}')
print(f'Normal safety gate:        >= {NORMAL_SAFETY_THRESHOLD}')
print(f'Bias declaration:           Always appended to Normal output')
"""))

# ── Step 10: Radiology card ───────────────────────────────────────────────────
CELLS.append(md("## Step 10 — Format the Report as a Clinical Card"))

CELLS.append(code("""\
from IPython.display import HTML, display

URGENCY_COLOURS = {
    'NORMAL':          ('#1a5c3a', '#e6f4eb'),
    'Normal':          ('#1a5c3a', '#e6f4eb'),
    'PNEUMONIA':       ('#8b1a1a', '#fce8e8'),
    'Pneumonia':       ('#8b1a1a', '#fce8e8'),
    'COVID-19':        ('#7a5200', '#fff3cd'),
    'Pleural Effusion':('#003e74', '#e8eef5'),
    'Indeterminate':   ('#4e5d7a', '#f2ede4'),
}

def display_radiology_card(image_path_or_pil, title='Chest X-Ray Review'):
    \"\"\"Full pipeline with formatted HTML output.\"\"\"
    pred_class, confidence, full_findings, report = classify_and_report(
        image_path_or_pil, title
    )
    pred_class, alerts = apply_guardrails(pred_class, confidence, full_findings)

    fg, bg = URGENCY_COLOURS.get(pred_class, ('#1a1a2e', '#f2ede4'))
    border = '#8b1a1a' if alerts else fg

    alert_html = ''.join(
        f'<div style="background:#fce8e8;border-left:4px solid #8b1a1a;'
        f'padding:10px 14px;margin-bottom:8px;border-radius:0 4px 4px 0;'
        f'font-weight:700;color:#8b1a1a">{a}</div>'
        for a in alerts
    )

    sections   = [s.strip() for s in report.split('|')]
    labels_    = ['Report', 'Primary Finding', 'Secondary Findings', 'Clinical Recommendation']
    section_html = ''.join(
        f'<tr><td style="padding:8px 14px;font-weight:700;color:{fg};'
        f'white-space:nowrap;vertical-align:top;width:200px">'
        f'{labels_[i] if i < len(labels_) else ""}</td>'
        f'<td style="padding:8px 14px;color:#1a1a2e">{sec}</td></tr>'
        for i, sec in enumerate(sections)
    )

    top6 = sorted(full_findings.items(), key=lambda x: -x[1])[:6]
    bars = ''.join(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
        f'<span style="width:160px;font-size:12px;color:#4e5d7a">{p}</span>'
        f'<div style="flex:1;background:#e8eef5;border-radius:3px;height:14px">'
        f'<div style="width:{v*100:.1f}%;background:{fg};height:14px;border-radius:3px"></div>'
        f'</div>'
        f'<span style="width:40px;font-size:12px;font-weight:700;color:{fg}">{v:.2f}</span>'
        f'</div>'
        for p, v in top6
    )

    html = f'''
    <div style="border:2px solid {border};border-radius:8px;overflow:hidden;
                font-family:'Segoe UI',Calibri,Arial,sans-serif;max-width:860px">
      <div style="background:#002147;padding:14px 20px;display:flex;
                  align-items:center;justify-content:space-between">
        <div>
          <div style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;
                      color:#d4ae4a;margin-bottom:4px">
            Oxford Clinical AI &nbsp;|&nbsp; DenseNet-121 ({WEIGHTS})
          </div>
          <div style="font-size:18px;font-weight:700;color:#fff">{title}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:26px;font-weight:700;color:{fg};background:{bg};
                      padding:6px 16px;border-radius:4px">{pred_class}</div>
          <div style="font-size:12px;color:rgba(255,255,255,.55);margin-top:4px">
            Confidence {confidence*100:.1f}% &nbsp;|&nbsp; Threshold {THRESHOLD}
          </div>
        </div>
      </div>
      <div style="padding:14px 18px;background:#fff">
        {alert_html}
        <table style="width:100%;border-collapse:collapse;margin-bottom:12px">
          {section_html}
        </table>
        <div style="border-top:1px solid #d4d9e3;padding-top:12px">
          <div style="font-size:12px;font-weight:700;letter-spacing:.1em;
                      text-transform:uppercase;color:#4e5d7a;margin-bottom:8px">
            Top Pathology Scores
          </div>
          {bars}
        </div>
      </div>
      <div style="background:#f2ede4;padding:8px 18px;font-size:11px;color:#666;
                  border-top:1px solid #d4d9e3">
        For radiologist review only &mdash; not for autonomous clinical decision-making
        &nbsp;|&nbsp; TorchXRayVision DenseNet-121 + GPT-4o-mini
        &nbsp;|&nbsp; Oxford Clinical AI Hackathon
      </div>
    </div>'''

    display(HTML(html))
    return pred_class, confidence, alerts


print('OK display_radiology_card() ready.')
"""))

# ── Example inference ─────────────────────────────────────────────────────────
CELLS.append(md("### Example — run on first test image"))

CELLS.append(code("""\
example_pil_img = test_ds[0]['image']
example_image_path = 'temp_cxr_example.png'
example_pil_img.save(example_image_path)

print(f'Running inference for image: {example_image_path}')
display_radiology_card(example_image_path, title='Test Inference from HuggingFace Dataset')

# Also show GradCAM
display_gradcam(example_pil_img, target='Pneumonia')
"""))

# ── Step 11: Upload widget ────────────────────────────────────────────────────
CELLS.append(md("""\
## Step 11 — Live Image Upload App (Demo Day Interface)

Upload any chest X-ray (.jpg / .png) and get a full radiology card + GradCAM.
Make sure this widget is running and tested before judges arrive.
"""))

CELLS.append(code("""\
import ipywidgets as widgets
from IPython.display import display, clear_output

upload_btn  = widgets.FileUpload(accept='.jpg,.jpeg,.png', multiple=False,
                                  description='Upload CXR')
analyse_btn = widgets.Button(description='Analyse', button_style='primary',
                              icon='search',
                              layout=widgets.Layout(width='140px'))
gradcam_btn = widgets.Button(description='GradCAM', button_style='info',
                              icon='eye',
                              layout=widgets.Layout(width='140px'))
status      = widgets.Label('Upload a chest X-ray to begin.')
output_area = widgets.Output()
current_img = [None]

def on_analyse(b):
    with output_area:
        clear_output()
        if not upload_btn.value:
            status.value = 'Please upload an image first.'
            return
        content = list(upload_btn.value.values())[0]['content']
        from io import BytesIO
        pil_img = Image.open(BytesIO(content)).convert('L')
        current_img[0] = pil_img
        status.value = 'Analysing...'
        try:
            pred, conf, alerts = display_radiology_card(pil_img, title='Live Demo Upload')
            alert_count = len(alerts)
            status.value = (
                f'{pred} ({conf*100:.1f}%) | '
                f'{alert_count} alert(s) | '
                f'Threshold: {THRESHOLD}'
            )
        except Exception as e:
            status.value = f'Error: {e}'

def on_gradcam(b):
    with output_area:
        if current_img[0] is None:
            status.value = 'Analyse an image first.'
            return
        status.value = 'Computing GradCAM...'
        display_gradcam(current_img[0], target='Pneumonia')
        status.value = 'GradCAM complete. Verify heatmap attends to correct anatomy.'

analyse_btn.on_click(on_analyse)
gradcam_btn.on_click(on_gradcam)

print('=== Oxford Clinical AI Hackathon — Challenge 4: Chest X-Ray Classifier ===')
print('Supported formats: .jpg, .jpeg, .png')
display(widgets.VBox([
    widgets.HBox([upload_btn, analyse_btn, gradcam_btn]),
    status,
    output_area
]))
"""))

# ── Step 12: Model card ───────────────────────────────────────────────────────
CELLS.append(md("""\
## Step 12 — Model Card (Slide 2: Explainability)

### Architecture

| Field | Detail |
|---|---|
| Model | TorchXRayVision DenseNet-121 |
| Weights | `densenet121-res224-all` (or chosen set from Step 6b) |
| Architecture | DenseNet-121 with 121 dense layers and skip connections |
| Input | Greyscale X-ray, normalised to [-1024, 1024], resized 224×224 |
| Output | 18 pathology sigmoid scores (multi-label, not mutually exclusive) |
| Parameters | ~7 million |
| Inference speed | ~0.5s per image on CPU |

### Training Data Provenance

| Dataset | Images | Institution |
|---|---|---|
| NIH ChestX-ray14 | 112,000 | US National Institutes of Health |
| CheXpert | 224,000 | Stanford University Medical Center |
| MIMIC-CXR | 370,000 | MIT / Beth Israel Deaconess Medical Center |
| PadChest | 160,000 | Hospital San Juan, Spain |
| RSNA Pneumonia | 30,000 | Radiological Society of North America |
| **Total** | **896,000** | |

### XAI Method — GradCAM (Selvaraju et al. 2016)

Gradient-weighted Class Activation Mapping. Visualises which spatial regions of the X-ray
most strongly activate the class score of interest. Implemented on the final DenseNet dense block.
Radiologists should verify the heatmap corresponds to anatomically plausible findings.

- Works on any CNN without retraining (Selvaraju et al. 2016)
- Human study evidence: helps untrained users distinguish stronger from weaker models

### Known Limitations

1. **COVID-19 is not a labelled training class** — model detects radiological features, not the aetiology
2. **Systematic underdiagnosis** of female, Black, Hispanic, and Medicaid patients across all training datasets — all Normal outputs require human review *(Seyyed-Kalantari et al. 2021, Nature Medicine)*
3. **Population bias** — trained on US and European hospitals; may underperform on other populations
4. **Film type** — validated on PA (posteroanterior) films only; not for AP or mobile X-rays
5. **GPT-4o-mini** may generate plausible but clinically inaccurate statements — radiologist review required
6. **Threshold** selected empirically on test set; not externally validated
7. **GradCAM heatmaps** require radiologist interpretation — model attention is not a diagnosis
"""))

# ── Step 13: Governance ───────────────────────────────────────────────────────
CELLS.append(md("""\
## Step 13 — Governance (Slide 3)

### 1. Acceptable Use

**Appropriate contexts:**
- Preliminary pathology flagging to support radiologist triage
- Medical education and trainee self-assessment
- Research: feature extraction and baseline comparison

**Explicitly excluded:**
- Autonomous reporting without radiologist oversight
- Paediatric patients (not trained or validated on paediatric films)
- Emergency settings without human confirmation
- Populations not represented in training data without additional validation *(Seyyed-Kalantari et al. 2021)*

### 2. Data Privacy

- Model weights are open source; no patient data required to run inference
- Uploaded X-rays are processed locally in Colab; no images sent to external servers
- GPT-4o-mini receives **probability scores only** — no pixel data, no patient identifiers
- Any NHS deployment must satisfy NHS DSPT, GDPR Article 9 (special category health data), and local IG policy

### 3. Monitoring Outputs

- **Monthly:** Radiologist audit — 20 random predictions reviewed against ground truth
- **Stratified by sex AND ethnicity** *(Seyyed-Kalantari et al. 2021)* — not just overall accuracy
- **Quarterly:** Re-evaluation against fixed hold-out set; alert if AUC drops > 0.05
- **Trigger:** If pneumonia sensitivity falls below 0.80 in audit → suspend and investigate
- **Feedback:** Per-report "Was this classification correct?" captured and logged
- **Gap:** Demographic stratification may not be available in all deployment contexts — flag as monitoring gap

### 4. Compliance with Existing Legislation

- **MHRA SaMD:** Likely Class IIa or IIb — regulatory approval required before NHS clinical use
- **NHS AI Framework:** Human oversight preserved; model is advisory only; clinician is decision-maker
- **Royal College of Radiologists AI guidance:** AI outputs must be presented as decision support, not diagnosis
- **UK GDPR Article 22:** Automated decision-making with significant effects on persons requires human review

### 5. Ownership & Liability

- Current UK law places liability with the **clinician**, not the model developer or deploying institution
- Escalation pathway: AI output contradicts clinical impression → second radiologist opinion required (not AI re-run)
- NHS trusts deploying AI tools must verify their clinical negligence indemnity covers AI-assisted decisions

### Evidence Base

| Paper | Key Contribution |
|---|---|
| Rajpurkar et al. 2017 (CheXNet) | Recall ≥ 0.90 safety bar; DenseNet-121 architecture validation |
| Selvaraju et al. 2016 (GradCAM) | XAI method; human study evidence for clinical trust |
| Seyyed-Kalantari et al. 2021 (*Nature Medicine*) | Underdiagnosis bias in under-served populations across all training datasets |
"""))

# ── Assemble & write ──────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "name": "challenge4_imaging.ipynb",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "cells": []
}

for cell in CELLS:
    c = dict(cell)
    # Convert multi-line string source to list of lines (nbformat convention)
    if isinstance(c["source"], str):
        lines = c["source"].splitlines(keepends=True)
        c["source"] = lines
    notebook["cells"].append(c)

out = "challenge4_imaging.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Saved: {out}")
print("Upload this file to Colab: File > Upload notebook")
