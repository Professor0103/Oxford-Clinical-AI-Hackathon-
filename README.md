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
