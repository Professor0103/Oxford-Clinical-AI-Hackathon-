import io
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = APP_ROOT / "data" / "model_cache"
MPL_CACHE_DIR = APP_ROOT / "data" / "mpl_cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib
import numpy as np
import torch
import torchxrayvision as xrv
import torchvision.transforms as T
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None


if load_dotenv is not None:
    load_dotenv()


DEFAULT_WEIGHTS = os.getenv("CXR_WEIGHTS", "densenet121-res224-chex")
# Consolidated from successful_challenge_3.py tuned run (Step 5/6 threshold tuning).
DEFAULT_THRESHOLD = float(os.getenv("CXR_THRESHOLD", "0.486"))
CXR_CLASS_MODE = os.getenv("CXR_CLASS_MODE", "notebook_binary")
PNEUMONIA_W_PNEU = float(os.getenv("CXR_W_PNEUMONIA", "0.50"))
PNEUMONIA_W_CONS = float(os.getenv("CXR_W_CONSOLIDATION", "0.35"))
PNEUMONIA_W_INFL = float(os.getenv("CXR_W_INFILTRATION", "0.15"))
PNEUMONIA_HARD_FLAG_THRESHOLD = float(os.getenv("CXR_PNEUMONIA_FLAG_THRESHOLD", "0.20"))
UNCERTAINTY_THRESHOLD = float(os.getenv("CXR_UNCERTAINTY_THRESHOLD", "0.25"))
NORMAL_SAFETY_THRESHOLD = float(os.getenv("CXR_NORMAL_SAFETY_THRESHOLD", "0.70"))
NON_CXR_HEURISTIC_THRESHOLD = float(os.getenv("CXR_NON_CXR_HEURISTIC_THRESHOLD", "0.03"))
SIMILARITY_THRESHOLD = float(os.getenv("CXR_SIMILARITY_THRESHOLD", "0.02"))
LOG_LEVEL = os.getenv("CXR_LOG_LEVEL", "INFO").upper()
LOG_PROMPTS = os.getenv("CXR_LOG_PROMPTS", "false").strip().lower() in {"1", "true", "yes", "on"}

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("cxr-app")

RADIOLOGY_SYS = """
You are a consultant radiologist assisting an AI-supported chest X-ray review workflow.

CONTEXT
- You will receive:
  1) a dictionary of 18 pathology probability scores (0.000-1.000) from a pre-trained TorchXRayVision DenseNet-121 model, and
  2) the top mapped challenge-class prediction with composite confidence.
  3) a list of active clinical guardrail alerts.
- You are not directly viewing the image.
- These are model-output signals for decision support, not definitive radiological observations or final diagnoses.
- Your task is to convert the score pattern into a concise, clinician-readable structured report suitable for display in a radiology results card.
- When synthesizing, prioritize the most clinically relevant findings (e.g., Pneumonia, Consolidation, Effusion) and align with the top challenge-class prediction, while still noting other significant pathologies.

OUTPUT FORMAT
Return exactly one pipe-delimited line in this exact format:
CHEST X-RAY REPORT | PRIMARY FINDING | SECONDARY FINDINGS | CLINICAL SIGNIFICANCE | URGENCY | RECOMMENDATION

IMPORTANT
- The six returned fields must contain content only.
- Do not repeat the field names inside the output.
- For example, the first field should be the report summary itself, not the literal text "CHEST X-RAY REPORT".

CARD DISPLAY OPTIMIZATION
- Each section will be shown in its own row on a clinical report card.
- Write each section as a compact field entry, not as a long paragraph.
- Keep each field short, high-yield, and readable at a glance.
- Avoid repeating the same finding across multiple fields unless necessary for safety.
- Prefer noun-phrase or brief declarative style over lengthy prose.

SECTION RULES
1) CHEST X-RAY REPORT
- Provide a one-sentence overall summary of the model-supported radiographic pattern, synthesizing the top challenge-class prediction and significant pathologies. If no strong abnormality is supported, state that no high-confidence acute cardiopulmonary abnormality is identified from the model outputs.

2) PRIMARY FINDING
- State the single most important and clinically prominent supported abnormality. Use cautious language such as 'probable', 'possible', 'suspicious for', or 'no dominant abnormality identified'. This should align closely with the top challenge-class prediction if relevant. If there's a critical guardrail alert (e.g., Pediatric Cardiomegaly), incorporate it here as the primary finding.

3) SECONDARY FINDINGS
- Include only relevant secondary or borderline findings (scores from 0.20 to 0.40). Describe these as 'borderline', 'indeterminate', or 'requiring attention'. Limit to 2-3 most clinically relevant items. If none are relevant, say: 'No material secondary finding supported'. Also, incorporate any non-critical guardrail alerts here.

4) CLINICAL SIGNIFICANCE
- Briefly state why the pattern matters clinically. Focus on triage relevance, need for review, or whether the pattern is nonspecific/low confidence. Do not provide a definitive diagnosis. Explicitly mention the clinical implications of any active guardrail alerts.

5) URGENCY
- Output ONLY one of the following exact phrases:
  URGENT
  REVIEW REQUIRED
  ROUTINE
  NO IMMEDIATE AI-IDENTIFIED URGENT FINDING
  This should reflect the highest urgency indicated by the model's prediction or any active guardrail alerts.

6) RECOMMENDATION
- Give one brief, conservative next step. Appropriate language includes 'radiologist review', 'clinical correlation', and 'follow-up imaging if clinically indicated'. Do not recommend treatment or disposition decisions. Ensure recommendations align with any triggered guardrail alerts.

INTERPRETATION RULES
- Prioritize the highest-probability clinically relevant thoracic findings for synthesis, especially Pneumonia, Consolidation, and Effusion.
- Treat Pneumonia and Consolidation as related air-space abnormality signals.
- Treat Effusion as a pleural abnormality signal.
- Integrate the 'Top challenge-class prediction' with the 18-pathology scores for a coherent narrative.
- Explicitly incorporate information from 'Active Guardrail Alerts' into the report, especially in 'PRIMARY FINDING', 'SECONDARY FINDINGS', and 'CLINICAL SIGNIFICANCE' sections. Use the alert text directly or paraphrase it concisely.
- Do not overcall COVID-19; describe the radiographic pattern rather than asserting aetiology unless explicitly provided as the mapped class.
- Do not infer laterality, severity, chronicity, interval change, devices, or technical quality unless directly supported by the provided input.
- Do not mention low-probability incidental labels that would clutter the card.

SAFETY RULES
- Do not guess.
- Do not fabricate image details, measurements, history, comparisons, or symptoms.
- Do not say you can see the image.
- Do not provide a final diagnosis.
- If the pattern is weak, mixed, or uncertain, state that clearly and direct radiologist review.
- When active guardrail alerts are present, ensure the report tone is cautious and prioritizes human review.

STYLE TARGET
- Formal radiology tone.
- Concise and specific.
- Suitable for a consultant-facing demo card.
- No markdown, no bullets, no extra commentary, no extra delimiters.
"""

URGENCY_COLOURS = {
    "NORMAL": ("#2D6A4F", "#E7F4ED"),
    "Normal": ("#2D6A4F", "#E7F4ED"),
    "PNEUMONIA": ("#B83A2A", "#FBE9E7"),
    "Pneumonia": ("#B83A2A", "#FBE9E7"),
    "COVID-19": ("#9C6B00", "#FFF4DB"),
    "Pleural Effusion": ("#1F5E8A", "#EAF2F8"),
    "Indeterminate": ("#5A6D84", "#EEF2F6"),
    "Not CXR / Unrecognisable": ("#B00020", "#FFCDD2"),
    "Ambiguous Findings - Triage Review": ("#E67C00", "#FFF3CD"),
    "Pediatric CXR - Caution": ("#FF5722", "#FFCCBC"),
}


@dataclass
class AnalysisResult:
    prediction: str
    confidence: float
    class_scores: dict[str, float]
    findings: dict[str, float]
    report: str
    alerts: list[str]
    threshold: float
    report_mode: str


def can_use_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and OpenAI is not None


def supports_dicom() -> bool:
    return pydicom is not None


@lru_cache(maxsize=1)
def get_model() -> Any:
    model = xrv.models.DenseNet(weights=DEFAULT_WEIGHTS, cache_dir=str(MODEL_CACHE_DIR))
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_transform() -> Any:
    return T.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ]
    )


@lru_cache(maxsize=1)
def get_pathology_index_map() -> dict[str, int]:
    model = get_model()
    return {name.lower(): idx for idx, name in enumerate(model.pathologies)}


def get_required_pathology_idx(*candidate_names: str) -> int:
    index_map = get_pathology_index_map()
    for name in candidate_names:
        idx = index_map.get(name.lower())
        if idx is not None:
            return idx
    model = get_model()
    raise RuntimeError(
        f"None of required pathologies {candidate_names!r} were found. "
        f"Available pathologies: {list(model.pathologies)!r}"
    )


def get_optional_pathology_idx(*candidate_names: str) -> int | None:
    index_map = get_pathology_index_map()
    for name in candidate_names:
        idx = index_map.get(name.lower())
        if idx is not None:
            return idx
    return None


PNEUMONIA_IDX = get_required_pathology_idx("Pneumonia")
CONSOLID_IDX = get_optional_pathology_idx("Consolidation")
INFILTR_IDX = get_optional_pathology_idx("Infiltration", "Lung Opacity")
EFFUSION_IDX = get_optional_pathology_idx("Effusion", "Pleural Effusion")

if CONSOLID_IDX is None:
    logger.warning("Consolidation label missing for selected weights; scoring degrades gracefully.")
if INFILTR_IDX is None:
    logger.warning("Infiltration/Lung Opacity label missing; scoring degrades gracefully.")
if EFFUSION_IDX is None:
    logger.warning("Effusion label missing; Pleural Effusion score defaults to 0.0.")


def _normalise_to_uint8(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    array -= float(array.min())
    max_value = float(array.max())
    if max_value > 0:
        array /= max_value
    return (array * 255).clip(0, 255).astype(np.uint8)


def _dicom_bytes_to_pil(data: bytes) -> Image.Image:
    if pydicom is None:
        raise RuntimeError("DICOM support requires the optional dependency `pydicom`.")

    dataset = pydicom.dcmread(io.BytesIO(data))
    pixels = dataset.pixel_array.astype(np.float32)

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    return Image.fromarray(_normalise_to_uint8(pixels)).convert("L")


def _safe_pathology_name(name: str, index: int) -> str:
    clean = str(name).strip()
    return clean if clean else f"Unknown_{index}"


def load_pil_image(source: Any) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("L")

    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("L")

    if hasattr(source, "name") and str(getattr(source, "name", "")).lower().endswith(".dcm"):
        if hasattr(source, "getvalue"):
            return _dicom_bytes_to_pil(source.getvalue())
        if hasattr(source, "read"):
            return _dicom_bytes_to_pil(source.read())

    if hasattr(source, "read"):
        return Image.open(source).convert("L")

    source_path = str(source)
    if source_path.lower().endswith(".dcm"):
        if pydicom is None:
            raise RuntimeError("DICOM support requires the optional dependency `pydicom`.")
        dataset = pydicom.dcmread(source_path)
        pixels = dataset.pixel_array.astype(np.float32)
        if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = pixels.max() - pixels
        return Image.fromarray(_normalise_to_uint8(pixels)).convert("L")

    return Image.open(source).convert("L")


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    img = np.array(pil_img.convert("L"))
    img = xrv.datasets.normalize(img, 255)
    img = img[None, ...]
    img = get_transform()(img)
    return torch.from_numpy(img).unsqueeze(0)


def predict_challenge_class(
    probs: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> tuple[str, float, dict[str, float]]:
    # Weighted fusion from successful notebook run:
    # 0.50*Pneumonia + 0.35*Consolidation + 0.15*Infiltration.
    pneumonia_score = (
        PNEUMONIA_W_PNEU * float(probs[PNEUMONIA_IDX])
        + PNEUMONIA_W_CONS * (float(probs[CONSOLID_IDX]) if CONSOLID_IDX is not None else 0.0)
        + PNEUMONIA_W_INFL * (float(probs[INFILTR_IDX]) if INFILTR_IDX is not None else 0.0)
    )

    if CXR_CLASS_MODE == "notebook_binary":
        class_scores = {"PNEUMONIA": pneumonia_score}
        if pneumonia_score >= threshold:
            return "PNEUMONIA", pneumonia_score, class_scores
        return "NORMAL", 1.0 - pneumonia_score, class_scores

    covid_inputs = []
    if CONSOLID_IDX is not None:
        covid_inputs.append(float(probs[CONSOLID_IDX]))
    if INFILTR_IDX is not None:
        covid_inputs.append(float(probs[INFILTR_IDX]))
    covid_score = float(max(covid_inputs)) if covid_inputs else pneumonia_score
    effusion_score = float(probs[EFFUSION_IDX]) if EFFUSION_IDX is not None else 0.0

    class_scores = {
        "PNEUMONIA": pneumonia_score,
        "COVID-19": covid_score,
        "Pleural Effusion": effusion_score,
    }
    top_label, top_score = max(class_scores.items(), key=lambda item: item[1])

    if top_score < threshold:
        return "NORMAL", 1.0 - top_score, class_scores
    return top_label, top_score, class_scores


def _build_fallback_report(
    prediction: str, confidence: float, findings: dict[str, float], alerts: list[str] | None = None,
) -> str:
    ranked = sorted(findings.items(), key=lambda item: item[1], reverse=True)
    top_findings = ", ".join(f"{name} {score:.2f}" for name, score in ranked[:3])
    borderline = [name for name, score in ranked if 0.20 <= score <= 0.40][:3]
    secondary = ", ".join(borderline) if borderline else "No additional borderline findings."

    if prediction == "COVID-19":
        primary = "Radiological features may be consistent with COVID-19, but aetiology is unconfirmed."
    elif prediction == "NORMAL":
        primary = "No challenge-class pathology exceeded the operating threshold."
    else:
        primary = f"Highest concern is {prediction.lower()} with composite confidence {confidence:.2f}."

    if confidence >= 0.7:
        urgency = "ROUTINE"
        significance = "Pattern is relatively stronger but still requires radiologist confirmation."
    elif confidence >= 0.5:
        urgency = "REVIEW REQUIRED"
        significance = "Pattern is mixed or nonspecific and should be reviewed in clinical context."
    else:
        urgency = "NO IMMEDIATE AI-IDENTIFIED URGENT FINDING"
        significance = "No dominant high-confidence acute pattern is supported by the model outputs."

    recommendation = "Radiologist review required before clinical use."
    return (
        "CHEST X-RAY REPORT "
        f"| {primary} "
        f"| Top model findings: {top_findings}. Borderline findings: {secondary} "
        f"| {significance} "
        f"| {urgency} "
        f"| {recommendation}"
    )


def generate_report(
    prediction: str,
    confidence: float,
    findings: dict[str, float],
    alerts: list[str] | None = None,
) -> tuple[str, str]:
    if not can_use_openai():
        logger.info("OpenAI reporting disabled; using local fallback report")
        return _build_fallback_report(prediction, confidence, findings, alerts=alerts), "fallback"

    model_name = os.getenv("OPENAI_REPORT_MODEL", "gpt-4o-mini")
    top_findings = sorted(findings.items(), key=lambda item: item[1], reverse=True)[:3]
    logger.info(
        "OpenAI report request starting: model=%s prediction=%s confidence=%.3f top_findings=%s",
        model_name,
        prediction,
        confidence,
        ", ".join(f"{name}:{score:.3f}" for name, score in top_findings),
    )

    alert_text = "; ".join(alerts) if alerts else "None"
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = (
        f"18-pathology probability scores from DenseNet-121: {findings}\n"
        f"Top challenge-class prediction: {prediction} "
        f"(composite confidence {confidence:.3f}).\n"
        f"Active Guardrail Alerts: {alert_text}\n"
        "Generate a concise, cautious and structured radiology card report for clinician review."
    )
    if LOG_PROMPTS:
        logger.info("OpenAI report prompt: %s", prompt)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": RADIOLOGY_SYS},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        content = response.choices[0].message.content
        if content:
            content = normalise_report_fields(content)
            logger.info(
                "OpenAI report request succeeded: model=%s response_chars=%d",
                model_name,
                len(content),
            )
            return content, "openai"
        logger.warning("OpenAI report response was empty; falling back to local report")
    except Exception as exc:
        logger.exception("OpenAI report request failed; falling back to local report: %s", exc)

    return _build_fallback_report(prediction, confidence, findings, alerts=alerts), "fallback"


def normalise_report_fields(report: str) -> str:
    """Strip echoed section labels if the model repeats them in the response."""
    fields = [field.strip() for field in report.split("|")]
    if not fields:
        return report

    label_map = {
        0: "CHEST X-RAY REPORT",
        1: "PRIMARY FINDING",
        2: "SECONDARY FINDINGS",
        3: "CLINICAL SIGNIFICANCE",
        4: "URGENCY",
        5: "RECOMMENDATION",
    }

    cleaned: list[str] = []
    for idx, field in enumerate(fields):
        label = label_map.get(idx)
        if label:
            upper = field.upper()
            if upper == label:
                field = ""
            elif upper.startswith(label + ":"):
                field = field[len(label) + 1 :].strip()
            elif upper.startswith(label + " -"):
                field = field[len(label) + 2 :].strip()
        cleaned.append(field or "Not stated.")

    return " | ".join(cleaned)


def apply_guardrails(
    prediction: str,
    confidence: float,
    findings: dict[str, float],
    is_pediatric: bool = False,
) -> tuple[str, list[str]]:
    alerts: list[str] = []

    # --- Pediatric override (highest priority) ---
    if is_pediatric:
        alerts.insert(
            0,
            "CRITICAL ALERT - Pediatric patient identified. Model not trained on "
            "pediatric datasets; cannot provide reliable interpretation. "
            "Refer to paediatric radiologist.",
        )
        prediction = "Pediatric CXR - Caution"

    # --- Non-CXR / unrecognisable image ---
    max_overall_prob = max(findings.values()) if findings else 0.0
    if max_overall_prob < NON_CXR_HEURISTIC_THRESHOLD:
        alerts.insert(
            0,
            f"CRITICAL ALERT - Unrecognisable image pattern detected "
            f"(max pathology score {max_overall_prob:.2f}). "
            "This may not be a chest X-ray. Referral back to emergency triage recommended.",
        )
        return "Not CXR / Unrecognisable", alerts

    # --- Ambiguous findings (similar pathology scores) ---
    relevant_scores = [score for score in findings.values() if score > 0.10]
    if len(relevant_scores) > 1:
        similarity_std = float(np.std(relevant_scores))
        if similarity_std < SIMILARITY_THRESHOLD:
            alerts.insert(
                0,
                "CRITICAL ALERT - Ambiguous findings: multiple clinically relevant pathologies "
                f"show near-identical scores (std {similarity_std:.3f} < {SIMILARITY_THRESHOLD:.3f}). "
                "Triage review required.",
            )
            prediction = "Ambiguous Findings - Triage Review"

    # --- Pneumonia hard flag ---
    pneumonia_raw = max(findings.get("Pneumonia", 0.0), findings.get("Consolidation", 0.0))
    if pneumonia_raw >= PNEUMONIA_HARD_FLAG_THRESHOLD:
        alerts.append(
            f"EXPEDITED REVIEW - Pneumonia/Consolidation score {pneumonia_raw:.2f} "
            f">= {PNEUMONIA_HARD_FLAG_THRESHOLD:.2f}",
        )

    # --- Uncertainty flag ---
    if confidence < UNCERTAINTY_THRESHOLD:
        alerts.append(
            f"UNCERTAIN - Composite confidence {confidence:.2f} < {UNCERTAINTY_THRESHOLD:.2f}; "
            "senior radiologist review required.",
        )

    # --- Normal safety gate ---
    if prediction == "NORMAL" and confidence < NORMAL_SAFETY_THRESHOLD:
        prediction = "Indeterminate"
        alerts.append(
            f"Normal withheld - confidence {confidence:.2f} < {NORMAL_SAFETY_THRESHOLD:.2f} safety gate.",
        )

    # --- Borderline secondary findings ---
    borderline = [
        (name, score)
        for name, score in findings.items()
        if 0.20 <= score <= 0.40 and name not in {"Pneumonia", "Consolidation"}
    ]
    if borderline:
        summary = ", ".join(
            f"{name} ({score:.2f})" for name, score in sorted(borderline, key=lambda item: -item[1])
        )
        alerts.append(f"Borderline secondary findings: {summary}")

    # --- Bias notice ---
    if prediction in {"NORMAL", "Normal"}:
        alerts.append(
            'BIAS NOTICE - "Normal" outputs require human review due to known underdiagnosis risk '
            "in under-served populations.",
        )

    # --- COVID-19 limitation ---
    if prediction == "COVID-19":
        alerts.append(
            "LIMITATION - COVID-19 is not a labelled training class; treat this as a radiological pattern only.",
        )

    return prediction, alerts


def analyse_image(
    source: Any,
    threshold: float | None = None,
    is_pediatric: bool = False,
) -> AnalysisResult:
    model = get_model()
    pil_img = load_pil_image(source)
    img_tensor = preprocess_image(pil_img)
    effective_threshold = float(DEFAULT_THRESHOLD if threshold is None else threshold)

    with torch.no_grad():
        probs = torch.sigmoid(model(img_tensor))[0].numpy()

    findings = {
        _safe_pathology_name(name, idx): round(float(value), 3)
        for idx, (name, value) in enumerate(zip(model.pathologies, probs))
    }
    prediction, confidence, class_scores = predict_challenge_class(probs, threshold=effective_threshold)
    prediction, alerts = apply_guardrails(
        prediction, confidence, findings, is_pediatric=is_pediatric,
    )
    report, report_mode = generate_report(prediction, confidence, findings, alerts=alerts)

    return AnalysisResult(
        prediction=prediction,
        confidence=float(confidence),
        class_scores=class_scores,
        findings=findings,
        report=report,
        alerts=alerts,
        threshold=effective_threshold,
        report_mode=report_mode,
    )


def build_radiology_card(result: AnalysisResult, title: str = "Chest X-Ray Review") -> str:
    fg, bg = URGENCY_COLOURS.get(result.prediction, ("#1a1a2e", "#f2ede4"))
    border = "#B83A2A" if result.alerts else fg
    report_badge = "GPT report" if result.report_mode == "openai" else "Local fallback report"

    alert_html = "".join(
        (
            "<div style=\"background:#FBE9E7;border-left:4px solid #B83A2A;"
            "padding:10px 14px;margin-bottom:8px;border-radius:0 4px 4px 0;"
            f"font-weight:700;color:#B83A2A\">{alert}</div>"
        )
        for alert in result.alerts
    )

    sections = [section.strip() for section in result.report.split("|")]
    labels = [
        "Report",
        "Primary Finding",
        "Secondary Findings",
        "Clinical Significance",
        "Urgency",
        "Recommendation",
    ]
    section_html = "".join(
        (
            f"<tr><td style=\"padding:8px 14px;font-weight:700;color:{fg};"
            "white-space:nowrap;vertical-align:top;width:200px\">"
            f"{labels[index] if index < len(labels) else ''}</td>"
            f"<td style=\"padding:8px 14px;color:#1a1a2e\">{section}</td></tr>"
        )
        for index, section in enumerate(sections)
    )

    top6 = sorted(result.findings.items(), key=lambda item: item[1], reverse=True)[:6]
    bars = "".join(
        (
            "<div style=\"display:flex;align-items:center;gap:8px;margin-bottom:4px\">"
            f"<span style=\"width:160px;font-size:12px;color:#4e5d7a\">{pathology}</span>"
            "<div style=\"flex:1;background:#e8eef5;border-radius:3px;height:14px\">"
            f"<div style=\"width:{score * 100:.1f}%;background:{fg};height:14px;border-radius:3px\"></div>"
            "</div>"
            f"<span style=\"width:40px;font-size:12px;font-weight:700;color:{fg}\">{score:.2f}</span>"
            "</div>"
        )
        for pathology, score in top6
    )

    return f"""
    <div style="border:2px solid {border};border-radius:10px;overflow:hidden;
                font-family:'Segoe UI',Calibri,Arial,sans-serif;max-width:860px;
                box-shadow:0 12px 24px rgba(0,0,0,0.08)">
      <div style="background:#002147;padding:16px 20px;display:flex;
                  align-items:center;justify-content:space-between">
        <div>
          <div style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;
                      color:#C7A94F;margin-bottom:4px">
            Oxford Clinical AI | DenseNet-121 ({DEFAULT_WEIGHTS})
          </div>
          <div style="font-size:20px;font-weight:700;color:#fff">{title}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:12px;color:rgba(255,255,255,.75);margin-top:4px">
            Confidence {result.confidence * 100:.1f}% | Threshold {result.threshold:.2f}
          </div>
          <div style="font-size:11px;color:#A0B4CC;margin-top:6px">{report_badge}</div>
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
      <div style="background:#F3F0E8;padding:8px 18px;font-size:11px;color:#666;
                  border-top:1px solid #d4d9e3">
        For radiologist review only - not for autonomous clinical decision-making |
        Pixel data stays local to this app; only numeric scores are sent for GPT reporting when enabled
      </div>
    </div>
    """


def build_export_html(result: AnalysisResult, title: str = "Chest X-Ray Review") -> str:
    card_html = build_radiology_card(result, title=title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      margin: 24px;
      background: #f4f6f8;
      font-family: 'Segoe UI', Calibri, Arial, sans-serif;
    }}
  </style>
</head>
<body>
  {card_html}
</body>
</html>
"""


def build_pdf_summary(result: AnalysisResult, image: Image.Image, title: str = "Chest X-Ray Review") -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setTitle(title)

    page_margin = 32
    card_x = page_margin
    card_y = 34
    card_w = width - (page_margin * 2)
    card_h = height - 68

    fg_hex, bg_hex = URGENCY_COLOURS.get(result.prediction, ("#1a1a2e", "#f2ede4"))
    fg = colors.HexColor(fg_hex)
    bg = colors.HexColor(bg_hex)
    navy = colors.HexColor("#0b2a4a")
    gold = colors.HexColor("#d4ae4a")
    surface = colors.HexColor("#f7f4ec")
    soft = colors.HexColor("#e8eef5")
    body = colors.HexColor("#1f2d3d")
    muted = colors.HexColor("#5b6b7a")
    alert_bg = colors.HexColor("#fce8e8")
    alert_border = colors.HexColor("#8b1a1a")

    pdf.setFillColor(colors.white)
    pdf.roundRect(card_x, card_y, card_w, card_h, 12, fill=1, stroke=0)

    header_h = 92
    pdf.setFillColor(navy)
    pdf.roundRect(card_x, card_y + card_h - header_h, card_w, header_h, 12, fill=1, stroke=0)
    pdf.setFillColor(navy)
    pdf.rect(card_x, card_y + card_h - header_h, card_w, header_h - 12, fill=1, stroke=0)

    pdf.setFillColor(gold)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(card_x + 20, card_y + card_h - 22, "Oxford Clinical AI | DenseNet-121")

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(card_x + 20, card_y + card_h - 42, title[:70])

    badge_w = 150
    badge_h = 28
    badge_x = card_x + card_w - badge_w - 20
    badge_y = card_y + card_h - 48
    pdf.setFillColor(bg)
    pdf.roundRect(badge_x, badge_y, badge_w, badge_h, 6, fill=1, stroke=0)
    pdf.setFillColor(fg)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawCentredString(badge_x + badge_w / 2, badge_y + 8, result.prediction)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica", 9)
    meta = f"Confidence {result.confidence * 100:.1f}% | Threshold {result.threshold:.2f}"
    pdf.drawRightString(card_x + card_w - 20, card_y + card_h - 62, meta)
    report_badge = "GPT report" if result.report_mode == "openai" else "Local fallback report"
    pdf.drawRightString(card_x + card_w - 20, card_y + card_h - 76, report_badge)

    content_top = card_y + card_h - header_h - 18
    preview = image.copy().convert("L")
    preview.thumbnail((180, 180))
    image_x = card_x + 20
    image_y = content_top - preview.height
    pdf.drawImage(ImageReader(preview), image_x, image_y, width=preview.width, height=preview.height)

    details_x = image_x + 200
    details_top = content_top
    pdf.setFillColor(body)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(details_x, details_top, "Case Summary")
    pdf.setFont("Helvetica", 10)
    summary_lines = [
        f"Prediction: {result.prediction}",
        f"Confidence: {result.confidence * 100:.1f}%",
        f"Report mode: {'GPT' if result.report_mode == 'openai' else 'Fallback'}",
    ]
    for idx, line in enumerate(summary_lines):
        pdf.drawString(details_x, details_top - 18 - (idx * 14), line)

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(details_x, details_top - 68, "Top Pathologies")
    bar_y = details_top - 86
    for pathology, score in sorted(result.findings.items(), key=lambda item: item[1], reverse=True)[:5]:
        pdf.setFont("Helvetica", 9)
        pdf.setFillColor(muted)
        pdf.drawString(details_x, bar_y + 2, pathology[:24])
        pdf.setFillColor(soft)
        pdf.roundRect(details_x + 82, bar_y, 120, 10, 3, fill=1, stroke=0)
        pdf.setFillColor(fg)
        pdf.roundRect(details_x + 82, bar_y, 120 * max(0.0, min(score, 1.0)), 10, 3, fill=1, stroke=0)
        pdf.setFillColor(body)
        pdf.drawRightString(details_x + 218, bar_y + 2, f"{score:.2f}")
        bar_y -= 16

    y = min(image_y, bar_y) - 24

    alerts = result.alerts or ["No additional alerts."]
    pdf.setFillColor(body)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(card_x + 20, y, "Alerts")
    y -= 16
    for alert in alerts:
        wrapped = _wrap_text(alert, 95)
        block_h = 12 + (len(wrapped) * 12)
        pdf.setFillColor(alert_bg)
        pdf.roundRect(card_x + 20, y - block_h + 2, card_w - 40, block_h, 5, fill=1, stroke=0)
        pdf.setFillColor(alert_border)
        pdf.rect(card_x + 20, y - block_h + 2, 4, block_h, fill=1, stroke=0)
        pdf.setFillColor(alert_border)
        pdf.setFont("Helvetica-Bold", 9)
        text_y = y - 10
        for idx, line in enumerate(wrapped):
            pdf.drawString(card_x + 32, text_y - (idx * 11), line)
        y -= block_h + 8
        if y < 160:
            pdf.showPage()
            pdf.setFillColor(colors.white)
            pdf.rect(0, 0, width, height, fill=1, stroke=0)
            y = height - 48

    pdf.setFillColor(body)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(card_x + 20, y, "Structured Report")
    y -= 16

    labels = [
        "Report",
        "Primary Finding",
        "Secondary Findings",
        "Clinical Significance",
        "Urgency",
        "Recommendation",
    ]
    sections = [section.strip() for section in result.report.split("|")]
    for idx, section in enumerate(sections):
        wrapped = _wrap_text(section, 85)
        pdf.setFillColor(colors.HexColor("#eef3f8"))
        block_h = 12 + (len(wrapped) * 12)
        pdf.roundRect(card_x + 20, y - block_h + 2, card_w - 40, block_h, 5, fill=1, stroke=0)
        pdf.setFillColor(fg)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(card_x + 30, y - 10, labels[idx] if idx < len(labels) else "Section")
        pdf.setFillColor(body)
        pdf.setFont("Helvetica", 9)
        for line_idx, line in enumerate(wrapped):
            pdf.drawString(card_x + 150, y - 10 - (line_idx * 11), line)
        y -= block_h + 8
        if y < 120:
            pdf.showPage()
            pdf.setFillColor(colors.white)
            pdf.rect(0, 0, width, height, fill=1, stroke=0)
            y = height - 48

    footer_h = 24
    pdf.setFillColor(surface)
    pdf.rect(card_x, card_y, card_w, footer_h, fill=1, stroke=0)
    pdf.setFillColor(muted)
    pdf.setFont("Helvetica", 8)
    footer = (
        "For radiologist review only - not for autonomous clinical decision-making | "
        "Pixel data stays local to this app; only score vectors are used for GPT reporting when enabled"
    )
    pdf.drawString(card_x + 14, card_y + 8, footer[:145])

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_length + extra > width:
            lines.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += extra

    if current:
        lines.append(" ".join(current))
    return lines


def gradcam_heatmap(source: Any, target: str = "Pneumonia") -> Image.Image:
    model = get_model()
    pil_img = load_pil_image(source)
    img_tensor = preprocess_image(pil_img)

    if target not in model.pathologies:
        target = "Pneumonia"

    target_idx = model.pathologies.index(target)
    gradients: list[torch.Tensor] = []
    activations: list[torch.Tensor] = []

    def save_gradient(grad: torch.Tensor) -> None:
        gradients.append(grad)

    def save_activation(module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> None:
        del module, inputs
        activations.append(output)
        output.register_hook(save_gradient)

    handle = model.features.denseblock4.register_forward_hook(save_activation)
    output = torch.sigmoid(model(img_tensor))
    handle.remove()

    model.zero_grad()
    output[0, target_idx].backward(retain_graph=True)

    pooled_grads = gradients[0].mean(dim=[0, 2, 3])
    activation = activations[0][0]
    for idx in range(activation.shape[0]):
        activation[idx, :, :] *= pooled_grads[idx]

    heatmap = activation.mean(dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_img.size, Image.LANCZOS)
    ) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    original = np.array(pil_img)

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title(f"GradCAM - {target}")
    axes[1].axis("off")

    axes[2].imshow(original, cmap="gray")
    axes[2].imshow(heatmap_resized, cmap="jet", alpha=0.45)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer)


def startup_diagnostics() -> list[tuple[str, str]]:
    diagnostics = [
        ("DenseNet weights", DEFAULT_WEIGHTS),
        ("Class mode", CXR_CLASS_MODE),
        (
            "Pneumonia fusion",
            f"{PNEUMONIA_W_PNEU:.2f}/{PNEUMONIA_W_CONS:.2f}/{PNEUMONIA_W_INFL:.2f} (P/C/I)",
        ),
        ("OpenAI reporting", "enabled" if can_use_openai() else "fallback mode"),
        ("DICOM support", "enabled" if supports_dicom() else "missing pydicom"),
        ("Pediatric guardrail", "active"),
        ("Non-CXR detection", f"threshold < {NON_CXR_HEURISTIC_THRESHOLD}"),
    ]
    return diagnostics
