"""
Challenge 4 — Imaging: Demo Day Slide Generator
Run this in Colab after completing the notebook to generate a .pptx file.

Usage:
    !pip install -q python-pptx
    %run generate_slides.py

Fill in the RESULTS block below with your actual numbers before running.
"""

# ── FILL IN YOUR RESULTS HERE ────────────────────────────────────────────────
RESULTS = {
    "weight_set":          "densenet121-res224-all",   # chosen weight set
    "threshold":           0.30,                        # chosen threshold
    "overall_accuracy":    0.00,                        # e.g. 0.82 → 82%
    "pneumonia_recall":    0.00,                        # e.g. 0.91 → 91%
    "pneumonia_precision": 0.00,
    "test_images":         200,
}
# ─────────────────────────────────────────────────────────────────────────────

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import os

# ── Colour palette (Oxford Clinical AI branding from notebook) ────────────────
OXFORD_BLUE   = RGBColor(0x00, 0x21, 0x47)
GOLD          = RGBColor(0xD4, 0xAE, 0x4A)
WHITE         = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY    = RGBColor(0xF2, 0xED, 0xE4)
RED_ALERT     = RGBColor(0x8B, 0x1A, 0x1A)
DARK_TEXT     = RGBColor(0x1A, 0x1A, 0x2E)
SLATE         = RGBColor(0x4E, 0x5D, 0x7A)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


def add_textbox(slide, text, left, top, width, height,
                font_size=18, bold=False, color=DARK_TEXT,
                align=PP_ALIGN.LEFT, wrap=True):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return txb


def fill_background(slide, color):
    from pptx.util import Emu
    from pptx.oxml.ns import qn
    from lxml import etree
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def header_bar(slide, title, subtitle=None):
    """Dark Oxford Blue header bar across the top."""
    bar = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(0), Inches(0), SLIDE_W, Inches(1.35)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = OXFORD_BLUE
    bar.line.fill.background()

    add_textbox(slide, "Oxford Clinical AI  ·  Challenge 4 — Imaging",
                Inches(0.35), Inches(0.08), Inches(9), Inches(0.4),
                font_size=11, color=GOLD)

    add_textbox(slide, title,
                Inches(0.35), Inches(0.42), Inches(10), Inches(0.65),
                font_size=28, bold=True, color=WHITE)

    if subtitle:
        add_textbox(slide, subtitle,
                    Inches(0.35), Inches(1.05), Inches(12), Inches(0.32),
                    font_size=13, color=RGBColor(0xCC, 0xCC, 0xCC))


def footer(slide):
    add_textbox(
        slide,
        "For radiologist review only — not for autonomous clinical decision-making  "
        "·  TorchXRayVision DenseNet-121 + GPT-4o-mini",
        Inches(0), Inches(7.2), SLIDE_W, Inches(0.3),
        font_size=9, color=SLATE, align=PP_ALIGN.CENTER
    )


def section_box(slide, label, body_lines, left, top, width, height,
                label_color=OXFORD_BLUE, bg_color=None):
    """Labelled content box."""
    if bg_color:
        rect = slide.shapes.add_shape(1, left, top, width, height)
        rect.fill.solid()
        rect.fill.fore_color.rgb = bg_color
        rect.line.color.rgb = RGBColor(0xCC, 0xD4, 0xE0)

    add_textbox(slide, label, left + Inches(0.15), top + Inches(0.1),
                width - Inches(0.2), Inches(0.35),
                font_size=12, bold=True, color=label_color)

    body = "\n".join(body_lines)
    add_textbox(slide, body,
                left + Inches(0.15), top + Inches(0.45),
                width - Inches(0.2), height - Inches(0.55),
                font_size=11, color=DARK_TEXT)


# ── Slide 1 — Performance ─────────────────────────────────────────────────────
def slide_performance(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    fill_background(slide, LIGHT_GREY)
    header_bar(slide,
               "Slide 1 — Performance",
               "Weight set selection · Threshold tuning · Benchmark evaluation")
    footer(slide)

    acc  = f"{RESULTS['overall_accuracy']*100:.1f}%"  if RESULTS['overall_accuracy']  else "[run notebook]"
    rec  = f"{RESULTS['pneumonia_recall']*100:.1f}%"  if RESULTS['pneumonia_recall']  else "[run notebook]"
    prec = f"{RESULTS['pneumonia_precision']*100:.1f}%" if RESULTS['pneumonia_precision'] else "[run notebook]"

    # ── Left column: model selection ─────────────────────────────────────────
    section_box(slide, "Model Selection",
        [
            f"Architecture:   DenseNet-121 (TorchXRayVision)",
            f"Weight set:     {RESULTS['weight_set']}",
            f"Selected by:    Step 6b comparison — all 5 weight sets",
            f"Criterion:      Pneumonia recall ≥ 0.90  (CheXNet, Rajpurkar et al. 2017)",
            f"Threshold:      {RESULTS['threshold']}  (evidence-grounded — see sweep below)",
        ],
        Inches(0.3), Inches(1.5), Inches(4.1), Inches(2.0),
        bg_color=WHITE)

    # ── Centre column: results ────────────────────────────────────────────────
    section_box(slide, "Test Set Results  (n=200, HuggingFace CXR)",
        [
            f"Overall accuracy:          {acc}",
            f"Pneumonia sensitivity:     {rec}   ← safety-critical metric",
            f"Pneumonia precision:       {prec}",
            f"",
            f"Confusion matrix:          see confusion_matrix.png",
            f"Threshold sweep plot:      see threshold_sweep.png",
        ],
        Inches(4.55), Inches(1.5), Inches(4.4), Inches(2.0),
        bg_color=WHITE)

    # ── Right column: benchmark context ──────────────────────────────────────
    section_box(slide, "Benchmark Context",
        [
            "CheXNet (Rajpurkar et al. 2017):",
            "  DenseNet-121 on NIH ChestX-ray14",
            "  Exceeded average radiologist F1",
            "  Recall ≥ 0.90 = clinical safety bar",
            "",
            "Seyyed-Kalantari et al. 2021:",
            "  Same training datasets",
            "  Underdiagnosis in ♀, Black,",
            "  Hispanic, Medicaid patients",
            "  → all Normal outputs: human review",
        ],
        Inches(9.1), Inches(1.5), Inches(3.9), Inches(2.0),
        bg_color=WHITE)

    # ── Bottom: threshold sweep explanation ──────────────────────────────────
    section_box(slide, "Threshold Selection — Evidence-Grounded Operating Point",
        [
            "Swept thresholds 0.10 → 0.50 in 0.05 steps after weight set selection.",
            "Chose the highest threshold still achieving pneumonia recall ≥ 0.90  (CheXNet safety bar).",
            "Tradeoff: lower threshold → higher sensitivity, more false positives. Documented explicitly.",
            "Plot saved to threshold_sweep.png — included in appendix / shown on request.",
        ],
        Inches(0.3), Inches(3.6), Inches(12.7), Inches(1.3),
        bg_color=WHITE)

    # ── COVID-19 caveat ───────────────────────────────────────────────────────
    section_box(slide, "⚠  Known Limitation: COVID-19",
        [
            "COVID-19 is not a labelled training class (pre-pandemic data).",
            "Model detects radiological features (bilateral consolidation, ground-glass opacity) — not the aetiology.",
            "COVID-19 predictions presented as: 'radiological features consistent with COVID-19; aetiology unconfirmed'.",
        ],
        Inches(0.3), Inches(5.0), Inches(12.7), Inches(1.15),
        bg_color=RGBColor(0xFF, 0xF3, 0xCD))


# ── Slide 2 — Explainability ──────────────────────────────────────────────────
def slide_explainability(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_background(slide, LIGHT_GREY)
    header_bar(slide,
               "Slide 2 — Explainability",
               "Model card · GradCAM · Known limitations")
    footer(slide)

    # ── Architecture ─────────────────────────────────────────────────────────
    section_box(slide, "Architecture",
        [
            "Model:        TorchXRayVision DenseNet-121",
            "Weights:      densenet121-res224-all (or chosen set)",
            "Parameters:   ~7 million",
            "Input:        Greyscale X-ray, normalised [-1024, 1024], 224×224",
            "Output:       18 pathology sigmoid scores (multi-label)",
            "Inference:    ~0.5s per image on CPU — no GPU required",
        ],
        Inches(0.3), Inches(1.5), Inches(4.1), Inches(2.3),
        bg_color=WHITE)

    # ── Training data provenance ──────────────────────────────────────────────
    section_box(slide, "Training Data Provenance",
        [
            "NIH ChestX-ray14    112,000   US Nat. Institutes of Health",
            "CheXpert            224,000   Stanford University Medical Center",
            "MIMIC-CXR           370,000   MIT / Beth Israel Deaconess",
            "PadChest            160,000   Hospital San Juan, Spain",
            "RSNA Pneumonia       30,000   Radiological Society of N. America",
            "─────────────────────────────────────────────",
            "Total:              896,000   images",
        ],
        Inches(4.55), Inches(1.5), Inches(4.4), Inches(2.3),
        bg_color=WHITE)

    # ── XAI method ───────────────────────────────────────────────────────────
    section_box(slide, "XAI Method — GradCAM  (Selvaraju et al. 2016)",
        [
            "Gradients of target pathology class → final DenseNet dense block",
            "Produces heatmap: where did the model look?",
            "Works on any CNN without retraining (Selvaraju et al. 2016)",
            "Human study evidence: helps untrained users judge model quality",
            "",
            "Radiologist validates anatomical plausibility:",
            "  Pneumonia → lower lobe consolidation",
            "  Effusion  → costophrenic angle / fluid meniscus",
            "  Normal    → no focal region should dominate",
        ],
        Inches(8.7), Inches(1.5), Inches(4.3), Inches(2.3),
        bg_color=WHITE)

    # ── Known limitations ────────────────────────────────────────────────────
    section_box(slide, "Known Limitations",
        [
            "1.  COVID-19 not a labelled training class — radiological features only",
            "2.  Systematic underdiagnosis: ♀, Black, Hispanic, Medicaid patients  (Seyyed-Kalantari et al. 2021)",
            "3.  Trained on US/European hospital populations — may underperform on other populations",
            "4.  Validated on PA (posteroanterior) films only — not for AP or mobile X-rays",
            "5.  GPT-4o-mini reports may be plausible but clinically inaccurate — radiologist review required",
            "6.  Operating threshold selected empirically on test set — not externally validated",
            "7.  GradCAM heatmaps require radiologist interpretation — model attention ≠ diagnosis",
        ],
        Inches(0.3), Inches(3.9), Inches(12.7), Inches(2.35),
        bg_color=RGBColor(0xFC, 0xE8, 0xE8))


# ── Slide 3 — Governance ─────────────────────────────────────────────────────
def slide_governance(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_background(slide, LIGHT_GREY)
    header_bar(slide,
               "Slide 3 — Governance",
               "Acceptable use · Data privacy · Monitoring · Compliance · Liability")
    footer(slide)

    # ── Acceptable use ───────────────────────────────────────────────────────
    section_box(slide, "Acceptable Use",
        [
            "✓  Radiologist triage support (advisory only)",
            "✓  Medical education and trainee self-assessment",
            "✓  Research: feature extraction and baseline comparison",
            "✗  Autonomous reporting without radiologist oversight",
            "✗  Paediatric patients (not trained/validated)",
            "✗  Emergency settings without human confirmation",
            "✗  Populations not represented in training data without additional validation",
        ],
        Inches(0.3), Inches(1.5), Inches(4.1), Inches(2.8),
        bg_color=WHITE)

    # ── Data privacy ─────────────────────────────────────────────────────────
    section_box(slide, "Data Privacy",
        [
            "Pixel data processed locally in Colab — never sent externally",
            "GPT-4o-mini receives probability scores only — no pixel data,",
            "  no patient identifiers  (GDPR Article 9 compliant)",
            "TorchXRayVision weights are open source — no patient data required",
            "NHS deployment: NHS DSPT + GDPR Article 9 (special category",
            "  health data) + local IG policy required",
        ],
        Inches(4.55), Inches(1.5), Inches(4.1), Inches(2.8),
        bg_color=WHITE)

    # ── Compliance ───────────────────────────────────────────────────────────
    section_box(slide, "Compliance & Liability",
        [
            "MHRA SaMD: likely Class IIa/IIb — regulatory approval",
            "  required before NHS clinical use",
            "NHS AI Framework: human oversight preserved; model is advisory",
            "RCR AI guidance: output = decision support, not diagnosis",
            "UK GDPR Article 22: human review required for automated",
            "  decisions with significant effect on persons",
            "Liability: clinician is decision-maker under current UK law;",
            "  NHS trust must verify indemnity covers AI-assisted decisions",
        ],
        Inches(8.7), Inches(1.5), Inches(4.3), Inches(2.8),
        bg_color=WHITE)

    # ── Monitoring ───────────────────────────────────────────────────────────
    section_box(slide, "Monitoring & Drift Detection",
        [
            "Monthly:    Radiologist audit — 20 random predictions reviewed against ground truth",
            "            Stratified by sex AND ethnicity  (Seyyed-Kalantari et al. 2021 — not just overall accuracy)",
            "Quarterly:  Re-evaluation against fixed hold-out set; alert if AUC drops > 0.05",
            "Trigger:    If pneumonia sensitivity falls below 0.80 in audit → suspend and retrain",
            "Feedback:   Per-report 'Was this classification correct?' captured and logged",
            "Gap:        Demographic stratification may not be available in all deployment contexts — flag as monitoring gap",
        ],
        Inches(0.3), Inches(4.4), Inches(12.7), Inches(1.85),
        bg_color=WHITE)


# ── Build & save ─────────────────────────────────────────────────────────────
def build():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    slide_performance(prs)
    slide_explainability(prs)
    slide_governance(prs)

    out = "challenge4_slides.pptx"
    prs.save(out)
    print(f"Saved: {out}")
    print("Download from the Colab file browser (left panel > Files tab).")


if __name__ == "__main__":
    build()
