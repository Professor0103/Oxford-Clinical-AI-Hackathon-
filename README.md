# Chest X-Ray Review (Streamlit)

Hackathon / portfolio project: a small Streamlit app that runs **TorchXRayVision** (DenseNet-121) on a single chest X-ray, applies simple **clinical guardrails**, shows an optional **GradCAM** overlay, and builds a **radiology-style card** with HTML/PDF export. If `OPENAI_API_KEY` is set, the narrative report uses the OpenAI API; otherwise it uses a built-in deterministic template.

**Not for clinical use** — outputs are advisory and require qualified review.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # optional: add OPENAI_API_KEY
streamlit run web_app.py
```

Supported uploads: `.png`, `.jpg`, `.jpeg`, `.dcm`. Demo images can live under `data/demo_images/` or `data/test_data/` (see folders in-repo for layout).

## Layout

| Path | Role |
|------|------|
| `web_app.py` | Streamlit UI |
| `app/core.py` | Model, mapping, guardrails, GradCAM, exports |
| `challenge4_imaging.ipynb` | Notebook workflow / experiments |
| `generate_slides.py` | Optional deck generator (`challenge4_slides.pptx`) |

Design choices draw on public chest-X-ray AI work (e.g. CheXNet-style recall thinking, Grad-CAM explainability, and bias-aware messaging; see e.g. [Seyyed-Kalantari et al., 2021](https://doi.org/10.1038/s41591-021-01595-0)).
