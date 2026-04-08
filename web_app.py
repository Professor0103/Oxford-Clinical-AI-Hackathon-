import io
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from app.core import (
    analyse_image,
    build_export_html,
    build_pdf_summary,
    build_radiology_card,
    can_use_openai,
    gradcam_heatmap,
    load_pil_image,
    startup_diagnostics,
    supports_dicom,
)


st.set_page_config(page_title="Chest X-Ray Review", layout="wide")

PRIMARY_IMAGE_DIR = Path("data/test_data")
FALLBACK_IMAGE_DIR = Path("data/demo_images")
SUPPORTED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm"}


def list_demo_images() -> list[Path]:
    base_dir = PRIMARY_IMAGE_DIR if PRIMARY_IMAGE_DIR.exists() else FALLBACK_IMAGE_DIR
    if not base_dir.exists():
        return []
    return sorted(
        path.relative_to(base_dir)
        for path in base_dir.rglob("*")
        if path.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS
    )


def load_selected_image(
    uploaded_file: Any, selected_demo_name: str
) -> tuple[Image.Image | None, str | None]:
    if uploaded_file is not None:
        return load_pil_image(uploaded_file), uploaded_file.name

    if selected_demo_name and selected_demo_name != "None":
        base_dir = PRIMARY_IMAGE_DIR if PRIMARY_IMAGE_DIR.exists() else FALLBACK_IMAGE_DIR
        path = base_dir / selected_demo_name
        return load_pil_image(path), str(path.relative_to(base_dir))

    return None, None


st.markdown(
    """
    <style>
      .app-shell {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(160deg, #f7f4ec 0%, #eef3f8 100%);
        border: 1px solid #d8e0ea;
        margin-bottom: 1rem;
      }
      .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #08294a;
        margin-bottom: 0.3rem;
      }
      .hero-copy {
        color: #3d5066;
        max-width: 60rem;
        line-height: 1.5;
      }
      .info-chip {
        display: inline-block;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        background: #e8eef5;
        color: #143a5c;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-top: 0.35rem;
      }
    </style>
    <div class="app-shell">
      <div class="hero-title">Chest X-Ray Clinical Review</div>
      <div class="hero-copy">
        Upload a chest X-ray, run the TorchXRayVision DenseNet-121 pipeline, and generate the
        final radiology card with guardrails, top findings, and optional GradCAM.
      </div>
      <div>
        <span class="info-chip">Hackathon demo mode</span>
        <span class="info-chip">Human review required</span>
        <span class="info-chip">Single-image analysis</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Input")
    uploaded_file = st.file_uploader(
        "Upload chest X-ray",
        type=sorted(SUPPORTED_EXTENSIONS),
        help="Supported formats: PNG, JPG, JPEG, and DICOM (.dcm).",
    )
    demo_images = list_demo_images()
    selected_demo = st.selectbox(
        "Or choose a local demo image",
        options=["None", *[str(path) for path in demo_images]],
        index=0,
    )
    run_gradcam = st.checkbox("Generate GradCAM", value=True)
    st.caption(
        "Default sample source is `data/test_data/` (falls back to `data/demo_images/` if needed)."
    )

    st.subheader("Readiness")
    for label, value in startup_diagnostics():
        st.write(f"**{label}:** {value}")

    if not can_use_openai():
        st.warning(
            "OpenAI reporting is not configured. The app will use a deterministic local fallback report."
        )
    if not supports_dicom():
        st.info("DICOM uploads need `pydicom` installed. PNG and JPG inputs still work.")

image, image_name = load_selected_image(uploaded_file, selected_demo)

if image is None:
    st.info("Upload an X-ray or add images to `data/test_data/` (or `data/demo_images/`) to begin.")
    st.markdown(
        """
        **Suggested next step**

        Add representative files to `data/test_data/` so the app is presentation-ready even offline.
        """
    )
    st.stop()

summary_cols = st.columns([1.1, 1, 1], gap="medium")
summary_cols[0].metric("Image selected", image_name or "Uploaded file")
summary_cols[1].metric("Report mode", "GPT" if can_use_openai() else "Fallback")
summary_cols[2].metric("GradCAM", "On" if run_gradcam else "Off")

left_col, right_col = st.columns([0.88, 1.12], gap="large")

with left_col:
    st.subheader("Source Image")
    st.image(image, caption=image_name or "Uploaded image", width="stretch", clamp=True)

with st.spinner("Running model inference and building radiology card..."):
    result = analyse_image(image)

with right_col:
    st.subheader("Radiology Card")
    st.markdown(
        build_radiology_card(result, title=image_name or "Chest X-Ray Review"),
        unsafe_allow_html=True,
    )

export_col_1, export_col_2 = st.columns(2, gap="medium")
html_export = build_export_html(result, title=image_name or "Chest X-Ray Review")
pdf_export = build_pdf_summary(result, image, title=image_name or "Chest X-Ray Review")

with export_col_1:
    st.download_button(
        "Download Card HTML",
        data=html_export,
        file_name="radiology_card.html",
        mime="text/html",
        width="stretch",
    )

with export_col_2:
    st.download_button(
        "Download PDF Summary",
        data=pdf_export,
        file_name="radiology_summary.pdf",
        mime="application/pdf",
        width="stretch",
    )

if result.report_mode == "fallback":
    st.info(
        "Structured report is using the local fallback path. Set `OPENAI_API_KEY` in `.env` to enable GPT report generation."
    )

score_col, findings_col = st.columns([0.8, 1.2], gap="large")

with score_col:
    st.subheader("Challenge Class Scores")
    for label, score in result.class_scores.items():
        st.metric(label, f"{score:.2f}")

with findings_col:
    st.subheader("Top Pathologies")
    top_findings = sorted(result.findings.items(), key=lambda item: item[1], reverse=True)[:8]
    st.dataframe(
        [{"Pathology": name, "Score": score} for name, score in top_findings],
        width="stretch",
        hide_index=True,
    )

if run_gradcam:
    st.subheader("GradCAM")
    with st.spinner("Computing GradCAM heatmap..."):
        gradcam_image = gradcam_heatmap(image, target="Pneumonia")
    st.image(
        gradcam_image,
        caption="Radiologist should confirm the model is attending to clinically plausible anatomy.",
        width="stretch",
    )

with st.expander("Model Card and Governance Notes"):
    st.markdown(
        """
        - Model: TorchXRayVision DenseNet-121 chest X-ray classifier.
        - Inputs: grayscale chest X-ray images resized to 224x224 after center crop.
        - Outputs: 18 pathology sigmoid scores mapped to challenge classes.
        - Limitation: COVID-19 is not a labelled training class; predictions indicate pattern only.
        - Governance: all outputs are advisory and require radiologist review.
        - Privacy: pixel data stays local to the app; only score vectors are used for GPT reporting.
        """
    )
