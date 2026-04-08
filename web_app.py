from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from app.core import (
    AnalysisResult,
    DEFAULT_THRESHOLD,
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
LOGO_PATH = Path("Nemo-AI.png")
SUPPORTED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm"}


def get_base_dir() -> Path:
    if PRIMARY_IMAGE_DIR.exists():
        return PRIMARY_IMAGE_DIR
    return FALLBACK_IMAGE_DIR


def list_queue_images() -> list[Path]:
    base_dir = get_base_dir()
    if not base_dir.exists():
        return []
    return sorted(
        path.relative_to(base_dir)
        for path in base_dir.rglob("*")
        if path.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS
    )


def extract_sections(report: str) -> list[str]:
    return [section.strip() for section in report.split("|")]


if "queue_idx" not in st.session_state:
    st.session_state.queue_idx = 0
if "queue_results" not in st.session_state:
    st.session_state.queue_results = {}


st.markdown(
    """
    <style>
      :root {
        --oxford-navy: #002147;
        --oxford-gold: #C7A94F;
        --oxford-slate: #A0B4CC;
        --oxford-alert: #B83A2A;
        --panel-bg: #F7F8FB;
      }
      .app-shell {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(160deg, #F7F8FB 0%, #EEF3F9 100%);
        border: 1px solid #D6DFEB;
        margin-bottom: 1rem;
      }
      .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: var(--oxford-navy);
        margin-bottom: 0.3rem;
      }
      .hero-copy {
        color: #33495F;
        max-width: 60rem;
        line-height: 1.5;
      }
      .info-chip {
        display: inline-block;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        background: #E8EEF7;
        color: var(--oxford-navy);
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-top: 0.35rem;
        border: 1px solid #D5DFEC;
      }
      .nemo-brand {
        font-family: "JetBrains Mono", "Consolas", "Courier New", monospace;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--oxford-navy);
        letter-spacing: 0.04em;
        margin-top: 0.2rem;
        margin-left: 0.2rem;
      }
    </style>
    <div class="app-shell">
      <div class="hero-title">Chest X-Ray Clinical Review Queue</div>
      <div class="hero-copy">
        Process the full test queue quickly: navigate images, run one or all, and review card-quality outputs
        with guardrails, explainability, and export options.
      </div>
      <div>
        <span class="info-chip">Queue workflow</span>
        <span class="info-chip">Human review required</span>
        <span class="info-chip">Judge-ready exports</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

queue_images = list_queue_images()
base_dir = get_base_dir()

if queue_images and st.session_state.queue_idx >= len(queue_images):
    st.session_state.queue_idx = 0

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=110)
    st.markdown('<div class="nemo-brand">Nemo.AI</div>', unsafe_allow_html=True)
    st.subheader("Queue Controls")
    run_gradcam = st.checkbox("Generate GradCAM", value=True)
    decision_threshold = st.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(DEFAULT_THRESHOLD),
        step=0.01,
        help="Notebook-aligned pneumonia decision threshold.",
    )

    if queue_images:
        current_label = str(queue_images[st.session_state.queue_idx])
        selected_label = st.selectbox(
            "Queue image",
            options=[str(path) for path in queue_images],
            index=st.session_state.queue_idx,
        )
        if selected_label != current_label:
            st.session_state.queue_idx = [str(path) for path in queue_images].index(selected_label)

        nav_prev_col, nav_next_col = st.columns(2)
        if nav_prev_col.button("Prev", width="stretch"):
            st.session_state.queue_idx = (st.session_state.queue_idx - 1) % len(queue_images)
            st.rerun()
        if nav_next_col.button("Next", width="stretch"):
            st.session_state.queue_idx = (st.session_state.queue_idx + 1) % len(queue_images)
            st.rerun()

        run_current = st.button("Run Current", type="primary", width="stretch")
        run_all = st.button("Run All Queue", width="stretch")
    else:
        run_current = False
        run_all = False

    st.subheader("Readiness")
    for label, value in startup_diagnostics():
        st.write(f"**{label}:** {value}")

    if not can_use_openai():
        st.warning("OpenAI reporting is not configured. Local fallback reports will be used.")
    if not supports_dicom():
        st.info("DICOM uploads need `pydicom` installed. PNG and JPG inputs still work.")


if not queue_images:
    st.info("No images found in queue. Add files under `data/test_data/` or `data/demo_images/`.")
    st.stop()


def run_one(path_rel: Path) -> tuple[Image.Image, AnalysisResult]:
    image_path = base_dir / path_rel
    image = load_pil_image(image_path)
    result = analyse_image(image, threshold=decision_threshold)
    st.session_state.queue_results[str(path_rel)] = result
    return image, result


if run_all:
    progress = st.progress(0.0, text="Running full queue...")
    for idx, rel in enumerate(queue_images, start=1):
        run_one(rel)
        progress.progress(idx / len(queue_images), text=f"Processed {idx}/{len(queue_images)}")
    st.success(f"Queue complete: {len(queue_images)}/{len(queue_images)} images processed.")


current_rel = queue_images[st.session_state.queue_idx]
current_image = load_pil_image(base_dir / current_rel)
current_result: AnalysisResult | None = st.session_state.queue_results.get(str(current_rel))

if run_current:
    current_image, current_result = run_one(current_rel)

rows = []
for rel in queue_images:
    key = str(rel)
    result = st.session_state.queue_results.get(key)
    if result is None:
        rows.append(
            {
                "Image": key,
                "Status": "Pending",
                "Confidence": "-",
                "Urgency": "-",
            }
        )
    else:
        sections = extract_sections(result.report)
        urgency = sections[4] if len(sections) > 4 else "-"
        rows.append(
            {
                "Image": key,
                "Status": "Done",
                "Confidence": f"{result.confidence * 100:.1f}%",
                "Urgency": urgency,
            }
        )

done_count = sum(1 for row in rows if row["Status"] == "Done")
summary_cols = st.columns([1, 1, 1], gap="medium")
summary_cols[0].metric("Queue progress", f"{done_count}/{len(queue_images)}")
summary_cols[1].metric("Current image", str(current_rel))
summary_cols[2].metric("Report mode", "GPT" if can_use_openai() else "Fallback")

st.subheader("Queue Results")
st.dataframe(rows, width="stretch", hide_index=True)

left_col, right_col = st.columns([0.88, 1.12], gap="large")
with left_col:
    st.subheader("Source Image")
    st.image(current_image, caption=str(current_rel), width="stretch", clamp=True)

if current_result is None:
    with right_col:
        st.subheader("Radiology Card")
        st.info("This image is pending. Click `Run Current` or `Run All Queue` to generate results.")
    st.stop()

with right_col:
    st.subheader("Radiology Card")
    st.markdown(
        build_radiology_card(current_result, title=str(current_rel)),
        unsafe_allow_html=True,
    )

export_col_1, export_col_2 = st.columns(2, gap="medium")
html_export = build_export_html(current_result, title=str(current_rel))
pdf_export = build_pdf_summary(current_result, current_image, title=str(current_rel))

with export_col_1:
    st.download_button(
        "Download Card HTML",
        data=html_export,
        file_name=f"{current_rel.stem}_radiology_card.html",
        mime="text/html",
        width="stretch",
    )
with export_col_2:
    st.download_button(
        "Download PDF Summary",
        data=pdf_export,
        file_name=f"{current_rel.stem}_radiology_summary.pdf",
        mime="application/pdf",
        width="stretch",
    )

if current_result.report_mode == "fallback":
    st.info("This result used fallback reporting. Set `OPENAI_API_KEY` in `.env` for GPT reports.")

score_col, findings_col = st.columns([0.8, 1.2], gap="large")
with score_col:
    st.subheader("Challenge Class Scores")
    for label, score in current_result.class_scores.items():
        st.metric(label, f"{score:.2f}")

with findings_col:
    st.subheader("Top Pathologies")
    top_findings = sorted(current_result.findings.items(), key=lambda item: item[1], reverse=True)[:8]
    st.dataframe(
        [{"Pathology": name, "Score": score} for name, score in top_findings],
        width="stretch",
        hide_index=True,
    )

if run_gradcam:
    st.subheader("GradCAM")
    with st.spinner("Computing GradCAM heatmap..."):
        gradcam_image = gradcam_heatmap(current_image, target="Pneumonia")
    st.image(
        gradcam_image,
        caption="Radiologist should confirm the model attends to clinically plausible anatomy.",
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
