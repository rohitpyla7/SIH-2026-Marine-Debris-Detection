"""
PDF report generation module for acoustic inspection survey logs.
Constructs multi-page survey summaries using ReportLab.
"""

import io
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, KeepTogether, PageBreak
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


# Theme palette
OCEAN_DARK = colors.HexColor("#06141F")
OCEAN_MID = colors.HexColor("#0A2230")
CARD_BG = colors.HexColor("#0D2B38")
BORDER = colors.HexColor("#1B5263")
ACCENT = colors.HexColor("#19C3D1")
ACCENT2 = colors.HexColor("#4DD0E1")
TEXT_MAIN = colors.HexColor("#E8F7FA")
TEXT_SEC = colors.HexColor("#91B7C0")
SUCCESS = colors.HexColor("#38D39F")
WARNING_C = colors.HexColor("#F4C95D")
DANGER = colors.HexColor("#FF667A")
ANOMALY_C = colors.HexColor("#B78CFF")
WHITE = colors.white
BLACK = colors.black


def _numpy_to_rl_image(img_array: np.ndarray, max_width: float,
                        max_height: float) -> Optional[RLImage]:
    """Convert a numpy BGR image to a ReportLab Image flowable."""
    try:
        if img_array is None:
            return None
        if img_array.ndim == 3:
            rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        else:
            rgb = img_array
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        h_orig, w_orig = img_array.shape[:2]
        scale = min(max_width / w_orig, max_height / h_orig, 1.0)
        return RLImage(buf, width=w_orig * scale, height=h_orig * scale)
    except Exception as e:
        print(f"[PDF] Image conversion error: {e}")
        return None


def generate_pdf_report(
    result,
    mission_id: str = "MISSION-001",
    output_path: str = None,
) -> bytes:
    """Generate comprehensive PDF survey and target inspection log."""
    if not REPORTLAB_OK:
        raise ImportError("reportlab not installed. Run: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf if output_path is None else output_path,
        pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        title="Side-Scan Sonar Inspection Report",
        author="Automated Sonar Analysis Pipeline",
    )

    W, H = A4
    content_w = W - 3.6 * cm

    styles = getSampleStyleSheet()

    def style(name, **kwargs) -> ParagraphStyle:
        base = styles.get(name, styles["Normal"])
        return ParagraphStyle(
            f"{name}_{id(kwargs)}",
            parent=base,
            **kwargs
        )

    title_style = style("title", fontSize=18, leading=22, textColor=ACCENT, alignment=TA_CENTER)
    subtitle_style = style("sub", fontSize=10, leading=14, textColor=TEXT_SEC, alignment=TA_CENTER)
    h1_style = style("h1", fontSize=12, leading=16, textColor=ACCENT2)
    body_style = style("body", fontSize=9, leading=12, textColor=TEXT_MAIN)
    warn_style = style("warn", fontSize=8, leading=11, textColor=WARNING_C)
    caption_style = style("cap", fontSize=8, leading=10, textColor=TEXT_SEC, alignment=TA_CENTER)

    table_header = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), CARD_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), ACCENT),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [OCEAN_DARK, CARD_BG]),
        ("TEXTCOLOR", (0, 1), (-1, -1), TEXT_MAIN),
    ])

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    dets = result.detections
    q = result.quality
    ar = result.anomaly_result

    story = []

    # Title section
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("UNDERWATER SIDE-SCAN SONAR", title_style))
    story.append(Paragraph("Debris & Anomaly Inspection Report", title_style))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(f"Survey Mission: {mission_id} | Record: {result.image_id}", subtitle_style))
    story.append(Paragraph(f"Generated: {now_str}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT))
    story.append(Spacer(1, 0.3 * cm))

    # Section 1: Imagery
    story.append(Paragraph("1. Acoustic Imagery Overview", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.2 * cm))

    img_w = (content_w - 0.4 * cm) / 3
    img_h = img_w * 0.4

    imgs_row = []
    for arr, cap in [
        (result.raw_image, "RAW ACOUSTIC"),
        (result.preprocessed_image, "ENHANCED & FILTERED"),
        (result.annotated_image, "DETECTIONS & TARGETS"),
    ]:
        rl = _numpy_to_rl_image(arr, img_w, img_h)
        cell = [rl or Paragraph("Frame unavailable", body_style),
                Paragraph(cap, caption_style)]
        imgs_row.append(cell)

    if any(c[0] for c in imgs_row):
        t = Table([imgs_row], colWidths=[img_w] * 3)
        t.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    # Section 2: Quality
    story.append(Paragraph("2. Signal Quality Analysis", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.2 * cm))

    metrics = q.get("metrics", {})
    q_data = [
        ["Metric", "Value"],
        ["Quality Grade", q.get("quality", "UNKNOWN")],
        ["Composite Score", f"{q.get('score', 0):.3f}"],
        ["Sharpness (Laplacian var.)", f"{metrics.get('sharpness', 0):.2f}"],
        ["Contrast Deviation", f"{metrics.get('contrast', 0):.2f}"],
        ["Acoustic Noise Index", f"{metrics.get('noise_estimate', 0):.2f}"],
        ["Mean Intensity", f"{metrics.get('mean_brightness', 0):.1f}"],
        ["Missing Data Columns", f"{metrics.get('missing_data_pct', 0):.1f}%"],
        ["Observations", "; ".join(q.get("reasons", [])) or "Nominal conditions"],
    ]
    qt = Table(q_data, colWidths=[content_w * 0.4, content_w * 0.6])
    qt.setStyle(table_header)
    story.append(qt)
    story.append(Spacer(1, 0.4 * cm))

    # Section 3: Performance
    story.append(Paragraph("3. Processing Latency Breakdown", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.2 * cm))

    timing = result.timing
    timing_data = [
        ["Pipeline Stage", "Duration (ms)"],
        ["Preprocessing & CLAHE", f"{timing.get('preprocess_ms', 0):.1f}"],
        ["Object Detection", f"{timing.get('detection_ms', 0):.1f}"],
        ["Anomaly Detection", f"{timing.get('anomaly_ms', 0):.1f}"],
        ["Acoustic Shadow Analysis", f"{timing.get('shadow_ms', 0):.1f}"],
        ["Total Latency", f"{timing.get('total_ms', 0):.1f}"],
    ]
    tt = Table(timing_data, colWidths=[content_w * 0.7, content_w * 0.3])
    tt.setStyle(TableStyle([
        *table_header._cmds,
        ("FONTNAME", (-1, -1), (-1, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (-1, -1), (-1, -1), colors.HexColor("#19C3D1")),
    ]))
    story.append(tt)
    story.append(Spacer(1, 0.4 * cm))

    # Section 4: Detections
    story.append(Paragraph(f"4. Target Detections ({len(dets)} target(s) identified)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.2 * cm))

    if not dets:
        story.append(Paragraph("No targets identified within threshold in this frame.", body_style))
    else:
        det_data = [["ID", "Class", "Confidence", "Evidence", "Shadow", "Anomaly", "Severity", "Classification"]]
        for det in dets:
            nat_type = "ARTIFICIAL" if not det.is_anomaly else "ANOMALY"
            det_data.append([
                det.detection_id[:12],
                det.display_name,
                f"{det.confidence:.0%}",
                f"{det.evidence_pct:.0f}%",
                f"{det.shadow_score:.0%}",
                f"{det.anomaly_score:.0%}",
                det.severity,
                nat_type,
            ])
        col_ws = [cm * 2.0, cm * 2.8, cm * 1.5, cm * 1.5, cm * 1.5, cm * 1.5, cm * 1.5, cm * 2.0]
        dt = Table(det_data, colWidths=col_ws)
        dt.setStyle(table_header)
        story.append(dt)

    story.append(Spacer(1, 0.4 * cm))

    # Section 5: Shadows
    story.append(Paragraph("5. Acoustic Shadow Validation", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.2 * cm))

    has_shadow = any(d.shadow_score > 0.1 for d in dets)
    if not has_shadow:
        story.append(Paragraph(
            "No prominent acoustic shadow signatures identified for targets in this frame.", body_style))
    else:
        for det in dets:
            if det.shadow_score > 0.1:
                sd = det.shadow_details
                shadow_data = [
                    ["Attribute", "Observation"],
                    ["Target", det.display_name],
                    ["Shadow Consistency", f"{sd.get('shadow_score', 0):.0%}"],
                    ["Target Intensity", f"{sd.get('object_intensity', 0):.1f}"],
                    ["Shadow Intensity", f"{sd.get('shadow_intensity', 0):.1f}"],
                    ["Seabed Context Intensity", f"{sd.get('background_intensity', 0):.1f}"],
                    ["Target Area (px)", f"{sd.get('object_area_px', 0):,}"],
                    ["Shadow Area (px)", f"{sd.get('shadow_area_px', 0):,}"],
                    ["Shadow/Target Ratio", f"{sd.get('shadow_to_object_ratio', 0):.2f}"],
                    ["Shadow Length (px)", f"{sd.get('shadow_length_px', 0)}"],
                    ["Height Extent Estimate", sd.get('height_estimate', 'Unavailable')[:80]],
                ]
                st_table = Table(shadow_data, colWidths=[content_w * 0.4, content_w * 0.6])
                st_table.setStyle(table_header)
                story.append(st_table)
                story.append(Spacer(1, 0.2 * cm))

    # Section 6: Anomaly
    story.append(Paragraph("6. Seabed Anomaly Scoring", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.2 * cm))

    anom_data = [
        ["Attribute", "Value"],
        ["Anomaly Index", f"{ar.get('anomaly_score', 0):.3f}"],
        ["Anomalous Status", str(ar.get("is_anomalous", False))],
        ["Evaluation Mode", ar.get("mode", "STANDARD")],
        ["Contributing Factors", "; ".join(ar.get("reasons", [])) or "Nominal background"],
    ]
    at = Table(anom_data, colWidths=[content_w * 0.35, content_w * 0.65])
    at.setStyle(table_header)
    story.append(at)
    story.append(Spacer(1, 0.4 * cm))

    # Section 7: Fusion
    story.append(Paragraph("7. Confidence Fusion Summary", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.2 * cm))

    if dets:
        fusion_data = [["Target", "Detector", "Shadow", "Anomaly", "Composite Evidence", "Severity"]]
        for det in dets:
            fusion_data.append([
                det.display_name,
                f"{det.confidence:.0%}",
                f"{det.shadow_score:.0%}",
                f"{det.anomaly_score:.0%}",
                f"{det.evidence_pct:.0f}%",
                det.severity,
            ])
        ft = Table(fusion_data, colWidths=[cm * 3.5, cm * 1.8, cm * 1.8, cm * 1.8, cm * 2.2, cm * 2.2])
        ft.setStyle(table_header)
        story.append(ft)

    story.append(Spacer(1, 0.4 * cm))
    story.append(PageBreak())

    # Section 8: Survey System Summary
    story.append(Paragraph("8. Inspection System Architecture", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.2 * cm))

    workflow_steps = [
        ("1. Input Ingestion", "Raw acoustic side-scan sonar image ingestion from survey datalog."),
        ("2. Quality Screening", "Contrast, Laplacian variance, and transducer dropout assessment."),
        ("3. Adaptive Equalization", "Contrast Limited Adaptive Histogram Equalization (CLAHE)."),
        ("4. Acoustic Denoising", "Bilateral / non-local means adaptive noise suppression."),
        ("5. Object Detection", "YOLOv8 deep neural inference with fallback heuristic rules."),
        ("6. Geomorphology Filter", "Differentiation between natural seabed features and debris."),
        ("7. Shadow Analysis", "Validation of physical elevation through acoustic shadow dropout."),
        ("8. Anomaly Detection", "Regional intensity and Shannon entropy deviation analysis."),
        ("9. Multi-Modal Fusion", "Cross-signal weighted evidence synthesis."),
        ("10. Threat Severity", "Operational risk categorization based on target dimensions and class."),
        ("11. Positional Tagging", "Synchronization with vehicle navigation and GPS telemetry."),
        ("12. Report Dispatch", "Tabular and spatial log generation for mission debrief."),
    ]
    wf_data = [["Step", "Pipeline Stage"]] + [[s, d] for s, d in workflow_steps]
    wft = Table(wf_data, colWidths=[cm * 4, content_w - cm * 4])
    wft.setStyle(table_header)
    story.append(wft)
    story.append(Spacer(1, 0.6 * cm))

    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
    story.append(Paragraph(
        f"Side-Scan Sonar Automated Target Inspection Log | {now_str}",
        caption_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
