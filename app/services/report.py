"""Generate a one-page PDF summary for a stored prediction."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as ReportImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.config import Settings
from app.models import Prediction
from app.services.inference import DISCLAIMER

PRIMARY = colors.HexColor("#7A1F55")
ACCENT = colors.HexColor("#E9C46A")
DARK = colors.HexColor("#1F2937")
LIGHT = colors.HexColor("#F6F3F5")


def build_report(prediction: Prediction, settings: Settings) -> bytes:
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"Breast Cancer Screening Report #{prediction.id}",
        author="Breast Cancer Detection using VGG19",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Times-Bold",
            fontSize=18,
            leading=22,
            textColor=PRIMARY,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=10.5,
            leading=14,
            textColor=DARK,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#4B5563"),
        )
    )

    created = prediction.created_at.strftime("%d %B %Y, %H:%M UTC") if isinstance(prediction.created_at, datetime) else str(prediction.created_at)
    story: list = [
        Paragraph("Breast Cancer Detection using VGG19", styles["ReportTitle"]),
        Paragraph("AI-assisted histopathological screening summary", styles["Small"]),
        Spacer(1, 4 * mm),
    ]

    summary_rows = [
        ["Report ID", f"BCD-{prediction.id:05d}"],
        ["Generated", created],
        ["Input image", prediction.filename],
        ["Image size", f"{prediction.image_width} x {prediction.image_height} px"],
        ["Model", f"VGG19 (transfer learning, {prediction.model_mode} backbone)"],
        ["Decision threshold", f"{prediction.threshold:.2f}"],
    ]
    summary = Table(summary_rows, colWidths=[38 * mm, 110 * mm])
    summary.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Times-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("TEXTCOLOR", (0, 0), (-1, -1), DARK),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#E5E7EB")),
            ]
        )
    )
    story += [summary, Spacer(1, 5 * mm)]

    outcome = (
        "MALIGNANT - suspicious pattern"
        if prediction.predicted_label == "malignant"
        else "BENIGN - no suspicious pattern detected"
    )
    outcome_colour = "#B3261E" if prediction.predicted_label == "malignant" else "#1B7F5A"
    banner = Table([[Paragraph(f"<b>{outcome}</b>", ParagraphStyle(
        name="Outcome", fontName="Times-Bold", fontSize=13, leading=16, textColor=colors.white
    ))]], colWidths=[148 * mm])
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(outcome_colour)),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story += [banner, Spacer(1, 4 * mm)]

    probability_rows = [
        ["Class", "Probability", "Risk level"],
        ["Benign", f"{prediction.benign_probability * 100:.1f} %", "-"],
        ["Malignant", f"{prediction.malignant_probability * 100:.1f} %", prediction.risk_level.capitalize()],
    ]
    probability_table = Table(probability_rows, colWidths=[40 * mm, 40 * mm, 40 * mm])
    probability_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story += [probability_table, Spacer(1, 5 * mm)]

    images = []
    original_path = Path(prediction.stored_image)
    if original_path.is_file():
        images.append(_image_cell(original_path, "Input histopathology patch"))
    if prediction.grad_cam_image and Path(prediction.grad_cam_image).is_file():
        images.append(_image_cell(Path(prediction.grad_cam_image), "Grad-CAM attention overlay"))
    if images:
        gallery = Table([images], colWidths=[74 * mm] * len(images))
        gallery.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story += [gallery, Spacer(1, 4 * mm)]

    if prediction.notes:
        story += [Paragraph("<b>Notes</b>", styles["Body"]), Paragraph(prediction.notes, styles["Body"]), Spacer(1, 3 * mm)]

    story += [
        Paragraph("<b>Interpretation guidance</b>", styles["Body"]),
        Paragraph(
            "The model analyses 224 x 224 pixel patches of H&amp;E stained breast tissue and "
            "estimates the probability of a malignant pattern. Grad-CAM highlights the image "
            "regions that contributed most to the decision. A low probability does not exclude "
            "disease, and a high probability is not a confirmed diagnosis.",
            styles["Body"],
        ),
        Spacer(1, 4 * mm),
        Paragraph("<b>Disclaimer</b>", styles["Body"]),
        Paragraph(DISCLAIMER, styles["Small"]),
        Spacer(1, 8 * mm),
        Paragraph("Report generated automatically by the Breast Cancer Detection web application.", styles["Small"]),
    ]

    document.build(story)
    return buffer.getvalue()


def _image_cell(path: Path, caption: str) -> list:
    from PIL import Image as PILImage

    with PILImage.open(path) as source:
        ratio = source.width / max(source.height, 1)
    width = 72 * mm
    height = width / ratio
    if height > 68 * mm:
        height = 68 * mm
        width = height * ratio
    image = ReportImage(str(path), width=width, height=height)
    style = ParagraphStyle(
        name="Caption",
        fontName="Times-Italic",
        fontSize=8.5,
        leading=10.5,
        textColor=colors.HexColor("#4B5563"),
    )
    return [image, Spacer(1, 1.5 * mm), Paragraph(caption, style)]
