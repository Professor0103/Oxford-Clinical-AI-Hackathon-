# Chest X-Ray Web App

This repo now includes a Streamlit web app version of the imaging workflow.

## What it does

- Uploads a local chest X-ray image
- Supports `.png`, `.jpg`, `.jpeg`, and `.dcm`
- Runs TorchXRayVision DenseNet-121 inference
- Maps outputs to challenge classes
- Applies clinical guardrails
- Produces the final radiology card
- Optionally renders a GradCAM overlay
- Exports the result as downloadable HTML and PDF
- Uses OpenAI for the structured report when `OPENAI_API_KEY` is set
- Falls back to a deterministic local report when no API key is available

## Local demo images

For a stable demo-day setup, place example `.png`, `.jpg`, `.jpeg`, or `.dcm` files in `data/demo_images/`.
That avoids pulling Step 2 images from the web at runtime.

I cannot safely invent clinical sample images here, so the repo includes the folder and workflow, but you should add 2-3 real demo-safe X-rays yourself.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Optionally copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
4. Start the app:

```powershell
streamlit run web_app.py
```

## Demo readiness checklist

- Add local sample images to `data/demo_images/`
- Confirm first model load works on the presentation machine
- If using GPT reports, set `OPENAI_API_KEY` in `.env`
- If using DICOM files, confirm `pydicom` is installed
- Keep a PNG or JPG backup in case a DICOM file is malformed

## Terminal logging

- `CXR_LOG_LEVEL=INFO` shows request start/success/fallback logs
- `CXR_LOG_LEVEL=DEBUG` is more verbose
- `CXR_LOG_PROMPTS=true` logs the full OpenAI report prompt to the terminal

## Notes

- The app is designed for single-image inference, not dataset benchmarking.
- All outputs are advisory and intended for radiologist review only.
- The app loads `.env` automatically when `python-dotenv` is installed.

## Paper-to-Code Traceability

| Paper | Implementation impact | Where it appears |
|---|---|---|
| Rajpurkar et al. 2017 (CheXNet) | DenseNet-121 as the core chest X-ray architecture; pneumonia recall `>= 0.90` used as the safety bar; threshold sweep and weight-set comparison framed around pneumonia sensitivity | `generate_notebook.py`, `challenge4_imaging.ipynb`, `ARCHITECTURE.md` |
| Selvaraju et al. 2016 (Grad-CAM) | GradCAM heatmap generation and overlay used as the explainability method for model-attention review | `app/core.py`, `generate_notebook.py`, `ARCHITECTURE.md` |
| Seyyed-Kalantari et al. 2021 | Bias/governance guardrails for under-served populations; explicit caution around Normal outputs; human-review-first messaging | `app/core.py`, `generate_notebook.py`, `ARCHITECTURE.md` |

### Practical mapping

- **CheXNet-inspired changes**:
  - DenseNet-121 model path
  - pneumonia-focused thresholding and evaluation logic
  - safety-bar language in notebook benchmarking
- **Grad-CAM-inspired changes**:
  - explainability visuals in notebook and app
  - radiologist validation of anatomical plausibility
- **Seyyed-Kalantari-inspired changes**:
  - bias notice and human-review guardrails
  - governance wording around underdiagnosis risk
  - caution against over-trusting apparently normal outputs

### Not directly paper-derived

These parts are implementation and demo engineering rather than direct research-paper translations:

- Streamlit web app UI
- HTML/PDF export
- local demo image store
- DICOM support
- model/log/cache/runtime setup
- GitHub/repo packaging
