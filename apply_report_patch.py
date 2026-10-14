"""
apply_report_patch.py - gives the PDF report a new, more engaging structure and wording.

It replaces ONLY the report code inside your own app.py (build_report_pdf, plus one small
new helper). Everything else is left exactly as it is, and the result page is unchanged.
A backup is saved first as app.py.bak.

Run from the project folder:
    python apply_report_patch.py
Then restart the app:  python app.py
"""
import ast
import os
import shutil
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "app.py"
MARK = "# --- report layout v2 ---"

NEW_REPORT = r'''# --- report layout v2 ---
def _report_wording(record, patient, summary):
    """Wording for the PDF only: indication, technique, findings, impression and
    priority, written in clinical-report style. The result page is unchanged."""
    cond = CONDITION_NAME or "the target condition"
    if len(cond) > 1 and cond[0].isupper() and cond[1].islower():
        cond = cond[0].lower() + cond[1:]          # "Diabetic foot ulcer" -> "diabetic foot ulcer"
    outcome = summary["outcome"]
    conf = record.confidence * 100
    level = summary["level"]

    # Indication: who was screened, and why
    bits = []
    if patient.age:
        bits.append(f"{patient.age}-year-old")
    if patient.gender:
        bits.append(str(patient.gender).lower())
    person = " ".join(bits) if bits else "patient"
    dm = []
    if patient.diabetes_type:
        dm.append(f"{patient.diabetes_type} diabetes")
    if patient.duration_years:
        dm.append(f"of {patient.duration_years:g} years' duration")
    with_dm = (" with " + " ".join(dm)) if dm else ""
    indication = (f"Screening for {cond} in a {person}{with_dm}. "
                  f"A digital clinical photograph was submitted for AI-assisted assessment.")

    technique = ("The photograph was analysed by an EfficientNet-based deep convolutional neural network. "
                 "A saliency map (gradient-weighted class activation mapping) was generated to show which "
                 "image regions most influenced the result.")

    ranked = sorted(record.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    others = ", ".join(f"{k} {v * 100:.1f}%" for k, v in ranked if k != record.prediction_label)
    findings = [f"Predicted category: <b>{record.prediction_label}</b>, with a model probability of "
                f"<b>{conf:.1f}%</b> ({level} confidence)."]
    if others:
        findings.append(f"Remaining categories: {others}.")
    if record.gradcam_filename:
        findings.append("The saliency map (Figure 2) marks the regions of greatest model attention. "
                        "Please correlate these with the lesion seen on examination.")
    else:
        findings.append("A saliency map is not available for this analysis.")

    # Impression + priority (low confidence always takes precedence)
    if conf < 60:
        word, headline = "INDETERMINATE", "INDETERMINATE"
        text = ("Model confidence is low, so this result is indeterminate. Expert review and repeat "
                "imaging are advised before any conclusion is drawn.")
        priority, tone = "MANUAL EXPERT REVIEW REQUIRED", "amber"
    elif outcome == "Affected":
        word, headline = "AFFECTED", "SCREEN-POSITIVE"
        text = (f"Image features are suggestive of {cond}. Clinical correlation and confirmatory "
                f"assessment are recommended.")
        priority, tone = "CLINICAL REVIEW ADVISED", "red"
    elif outcome == "Not affected":
        word, headline = "NOT AFFECTED", "SCREEN-NEGATIVE"
        text = (f"No image features suggestive of {cond} were identified. A negative screen does not "
                f"exclude disease, so please correlate clinically.")
        priority, tone = "ROUTINE FOLLOW-UP", "green"
    else:
        word, headline = str(record.prediction_label).upper()[:16], "RESULT"
        text = (f"The model classified this image as '{record.prediction_label}'. Its clinical meaning has "
                f"not been configured, so interpretation rests with the treating clinician.")
        priority, tone = "INTERPRETATION BY CLINICIAN", "purple"

    return {"indication": indication, "technique": technique, "findings": findings, "word": word,
            "headline": headline, "impression": text, "priority": priority, "tone": tone}


def build_report_pdf(record, patient, filepath):
    """Builds the clinical-style PDF screening report and writes it to filepath."""
    from xml.sax.saxutils import escape as esc
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                                     Table, TableStyle, KeepTogether, HRFlowable, Image as RLImage)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.graphics.shapes import Drawing, Rect, String, Wedge
    from PIL import Image as PILImage
    import numpy as np

    summary = clinical_summary(record)
    words = _report_wording(record, patient, summary)

    PRIMARY = colors.HexColor("#5b21b6")
    ACCENT = colors.HexColor("#8b5cf6")
    LIGHT = colors.HexColor("#f5f3ff")
    TEXT = colors.HexColor("#1f2937")
    MUTED = colors.HexColor("#6b7280")
    BORDER = colors.HexColor("#e5e7eb")
    TONES = {"red": ("#b91c1c", "#fef2f2"), "green": ("#15803d", "#f0fdf4"),
             "amber": ("#b45309", "#fffbeb"), "purple": ("#5b21b6", "#f5f3ff")}
    sev = colors.HexColor(TONES[words["tone"]][0])
    sev_tint = colors.HexColor(TONES[words["tone"]][1])

    W, H = A4
    LM = RM = 18 * mm
    CW = W - LM - RM

    lbl = ParagraphStyle("lbl", fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=MUTED)
    val = ParagraphStyle("val", fontName="Helvetica", fontSize=9.5, leading=12, textColor=TEXT)
    valb = ParagraphStyle("valb", parent=val, fontName="Helvetica-Bold")
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=PRIMARY, spaceBefore=9)
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.2, leading=13.2, textColor=TEXT)
    small = ParagraphStyle("small", fontName="Helvetica", fontSize=7.8, leading=10.5, textColor=MUTED)
    cap = ParagraphStyle("cap", parent=small, alignment=TA_CENTER)
    bul = ParagraphStyle("bul", parent=body, leftIndent=13, firstLineIndent=-13, spaceAfter=3.5)
    sbul = ParagraphStyle("sbul", parent=small, leftIndent=9, firstLineIndent=-9, spaceAfter=2)
    fbul = ParagraphStyle("fbul", parent=body, leftIndent=10, firstLineIndent=-10, spaceAfter=2)
    title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=TEXT)
    sub = ParagraphStyle("sub", fontName="Helvetica", fontSize=9.5, leading=13, textColor=MUTED)
    head = ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=sev)
    imp = ParagraphStyle("imp", fontName="Helvetica-Bold", fontSize=10.5, leading=14.5, textColor=TEXT)
    warn = ParagraphStyle("warn", parent=body, textColor=colors.HexColor("#b45309"), fontSize=8.6, leading=12)
    prio = ParagraphStyle("prio", fontName="Helvetica-Bold", fontSize=9.5, leading=12, textColor=sev)
    disc = ParagraphStyle("disc", fontName="Helvetica", fontSize=7.8, leading=11, textColor=colors.HexColor("#7f1d1d"))

    created = record.created_at.strftime("%d %b %Y, %H:%M UTC") if record.created_at else "N/A"
    report_no = f"DV-{record.id:04d}"
    cond_title = (CONDITION_NAME or "").strip()
    conf = record.confidence * 100
    level_text = ("High (85% or above)" if conf >= 85 else
                  "Moderate (60 to 85%)" if conf >= 60 else "Low (below 60%)")

    def section(t):
        return [Paragraph(t, h2), HRFlowable(width="100%", thickness=0.8, color=ACCENT, spaceBefore=1, spaceAfter=6)]

    def decorate(canv, doc):
        canv.saveState()
        canv.setFillColor(PRIMARY)
        canv.rect(0, H - 26 * mm, W, 26 * mm, stroke=0, fill=1)
        canv.setFillColor(ACCENT)
        canv.rect(0, H - 27.2 * mm, W, 1.2 * mm, stroke=0, fill=1)
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica-Bold", 19)
        canv.drawString(LM, H - 14 * mm, "DiaVision AI")
        canv.setFont("Helvetica", 8)
        canv.drawString(LM, H - 20 * mm, "INTELLIGENCE SUITE  |  AI-ASSISTED SCREENING REPORT")
        canv.setFont("Helvetica-Bold", 9.5)
        canv.drawRightString(W - RM, H - 13 * mm, f"Report No. {report_no}")
        canv.setFont("Helvetica", 8)
        canv.drawRightString(W - RM, H - 19 * mm, f"Generated: {created}")
        canv.setStrokeColor(BORDER)
        canv.setLineWidth(0.6)
        canv.line(LM, 15 * mm, W - RM, 15 * mm)
        canv.setFillColor(MUTED)
        canv.setFont("Helvetica", 7.5)
        canv.drawString(LM, 10.5 * mm, f"Confidential  |  Patient {patient.patient_code}  |  AI-generated decision support, not a final diagnosis.")
        canv.drawRightString(W - RM, 10.5 * mm, f"Page {doc.page}")
        canv.restoreState()

    story = [Paragraph("AI-Assisted Clinical Screening Report", title),
             Paragraph(esc(f"{cond_title} screening  |  Report {report_no}" if cond_title
                           else f"Clinical photograph screening  |  Report {report_no}"), sub),
             Spacer(1, 7)]

    # ---- 1. Result at a glance --------------------------------------------
    def verdict_drawing():
        w, h = 60 * mm, 30 * mm
        d = Drawing(w, h)
        d.add(Rect(0, 0, w, h, rx=3 * mm, ry=3 * mm, fillColor=sev, strokeColor=None))
        d.add(String(5 * mm, h - 7.5 * mm, "AI SCREENING RESULT", fontName="Helvetica-Bold", fontSize=6.5,
                     fillColor=colors.Color(1, 1, 1, 0.85)))
        fs, avail = 21.0, w - 10 * mm
        while stringWidth(words["word"], "Helvetica-Bold", fs) > avail and fs > 8:
            fs -= 0.5
        d.add(String(5 * mm, 12.5 * mm, words["word"], fontName="Helvetica-Bold", fontSize=fs, fillColor=colors.white))
        subline = "LOW CONFIDENCE" if words["headline"] == "INDETERMINATE" else words["headline"]
        d.add(String(5 * mm, 5.5 * mm, subline, fontName="Helvetica", fontSize=7.5,
                     fillColor=colors.Color(1, 1, 1, 0.9)))
        return d

    def gauge_drawing():
        w, h = 40 * mm, 30 * mm
        cx, cy, r, r1 = w / 2, h / 2, 13 * mm, 9.3 * mm
        d = Drawing(w, h)
        d.add(Wedge(cx, cy, r, 0, 359.9, radius1=r1, fillColor=colors.HexColor("#ede9fe"), strokeColor=None))
        p = min(max(record.confidence, 0.004), 1.0)
        d.add(Wedge(cx, cy, r, 90 - 360 * p, 90, radius1=r1, fillColor=sev, strokeColor=None))
        d.add(String(cx, cy - 1.2 * mm, f"{conf:.1f}%", textAnchor="middle", fontName="Helvetica-Bold",
                     fontSize=10.5, fillColor=TEXT))
        d.add(String(cx, cy - 5.2 * mm, "CONFIDENCE", textAnchor="middle", fontName="Helvetica",
                     fontSize=4.8, fillColor=MUTED))
        return d

    facts = [Paragraph("PREDICTED CATEGORY", lbl), Paragraph(esc(record.prediction_label), valb), Spacer(1, 3),
             Paragraph("CONFIDENCE LEVEL", lbl), Paragraph(level_text, valb), Spacer(1, 3),
             Paragraph("REVIEWED BY", lbl), Paragraph(esc(record.reviewed_by or "Pending review"), valb)]
    glance = Table([[verdict_drawing(), gauge_drawing(), facts]],
                   colWidths=[62 * mm, 42 * mm, CW - 104 * mm])
    glance.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story += [glance, Spacer(1, 6)]

    strip = Table([[Paragraph(f"NEXT STEP:  {words['priority']}", prio)]], colWidths=[CW])
    strip.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), sev_tint),
        ("LINEBEFORE", (0, 0), (0, 0), 4, sev),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
    ]))
    story += [strip]

    # ---- 2. Patient & examination details ---------------------------------
    name = (patient.name or "").strip() or "Not provided"
    age_sex = " / ".join(str(x) for x in (patient.age, patient.gender) if x) or "N/A"
    dm_bits = [str(patient.diabetes_type)] if patient.diabetes_type else []
    if patient.duration_years:
        dm_bits.append(f"{patient.duration_years:g} years")
    diabetes_txt = "  (".join([dm_bits[0], dm_bits[1] + ")"]) if len(dm_bits) == 2 else (dm_bits[0] if dm_bits else "N/A")
    P = lambda t: Paragraph(esc(str(t)), val)
    L = lambda t: Paragraph(t, lbl)
    details = Table([
        [L("PATIENT NAME"), P(name), L("PATIENT ID"), P(patient.patient_code)],
        [L("AGE / SEX"), P(age_sex), L("DIABETES"), P(diabetes_txt)],
        [L("EXAMINATION DATE"), P(created), L("REPORT NO."), P(report_no)],
        [L("REVIEWING CLINICIAN"), P(record.reviewed_by or "Pending review"), L("IMAGE TYPE"),
         P("Clinical photograph (digital)")],
    ], colWidths=[33 * mm, CW / 2 - 33 * mm, 29 * mm, CW / 2 - 29 * mm])
    details.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += section("Patient & Examination Details") + [details]

    # ---- 3. Clinical report: indication, technique, findings ---------------
    rep = Table([
        [Paragraph("INDICATION", lbl), Paragraph(esc(words["indication"]), body)],
        [Paragraph("TECHNIQUE", lbl), Paragraph(esc(words["technique"]), body)],
        [Paragraph("FINDINGS", lbl), [Paragraph("\u2022  " + f, fbul) for f in words["findings"]]],
    ], colWidths=[27 * mm, CW - 27 * mm])
    rep.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, BORDER),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += section("Clinical Report") + [rep]

    # ---- 4. Impression -----------------------------------------------------
    cell = [Paragraph("IMPRESSION", lbl), Spacer(1, 2),
            Paragraph(words["headline"], head), Spacer(1, 2),
            Paragraph(esc(words["impression"]), imp)]
    if summary["caution"] and words["headline"] != "INDETERMINATE":
        cell += [Spacer(1, 4), Paragraph("<b>Note:</b> " + esc(summary["caution"]), warn)]
    box = Table([[cell]], colWidths=[CW])
    box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("LINEBEFORE", (0, 0), (0, 0), 4, sev),
        ("BACKGROUND", (0, 0), (-1, -1), sev_tint),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story += [KeepTogether(section("Impression") + [box])]

    # ---- 5. Model output ----------------------------------------------------
    BAR_W = 70 * mm

    def bar(prob, top):
        d = Drawing(BAR_W, 4 * mm)
        d.add(Rect(0, 0, BAR_W, 4 * mm, fillColor=colors.HexColor("#ede9fe"), strokeColor=None))
        d.add(Rect(0, 0, max(BAR_W * prob, 0.6), 4 * mm, fillColor=(sev if top else ACCENT), strokeColor=None))
        return d

    ranked = sorted(record.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    rows = [[Paragraph("CATEGORY", lbl), Paragraph("MODEL PROBABILITY", lbl), Paragraph("SCORE", lbl)]]
    for i, (label, prob) in enumerate(ranked):
        rows.append([Paragraph(f"<b>{esc(label)}</b>" if i == 0 else esc(label), val), bar(prob, i == 0),
                     Paragraph(f"<b>{prob * 100:.1f}%</b>" if i == 0 else f"{prob * 100:.1f}%", val)])
    pr = Table(rows, colWidths=[CW * 0.34, BAR_W + 6 * mm, CW - CW * 0.34 - BAR_W - 6 * mm])
    pr.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, BORDER),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [KeepTogether(section("Model Output") + [pr])]

    # ---- 6. Visual evidence --------------------------------------------------
    def fit_img(path, max_w, max_h):
        with PILImage.open(path) as im:
            w, h = im.size
        s = min(max_w / w, max_h / h)
        return RLImage(path, width=w * s, height=h * s)

    img_path = os.path.join(UPLOAD_FOLDER, record.image_filename)
    left = fit_img(img_path, 72 * mm, 54 * mm) if os.path.exists(img_path) else Paragraph("Image not available.", cap)
    right = Paragraph("Saliency map not available for this analysis.", cap)
    if record.gradcam_filename:
        gp = os.path.join(UPLOAD_FOLDER, record.gradcam_filename)
        if os.path.exists(gp):
            right = fit_img(gp, 72 * mm, 54 * mm)
    imgs = Table([[left, right],
                  [Paragraph("Figure 1. Submitted clinical photograph", cap),
                   Paragraph("Figure 2. Saliency map: warmer colours mark regions that most influenced the model", cap)]],
                 colWidths=[CW / 2, CW / 2])
    imgs.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
        ("BOX", (0, 0), (0, -1), 0.6, BORDER), ("BOX", (1, 0), (1, -1), 0.6, BORDER),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fafafa")),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    def legend_drawing():
        d = Drawing(CW, 7 * mm)
        n, bw = 60, 46 * mm
        x0 = (CW - bw) / 2
        cols = _jet_colormap(np.linspace(0, 1, n)[None, :])[0]
        for i in range(n):
            d.add(Rect(x0 + i * bw / n, 2 * mm, bw / n + 0.3, 2.6 * mm, strokeColor=None,
                       fillColor=colors.Color(cols[i][0] / 255, cols[i][1] / 255, cols[i][2] / 255)))
        d.add(String(x0 - 2 * mm, 2.4 * mm, "Low influence", textAnchor="end", fontName="Helvetica",
                     fontSize=6.8, fillColor=MUTED))
        d.add(String(x0 + bw + 2 * mm, 2.4 * mm, "High influence", fontName="Helvetica",
                     fontSize=6.8, fillColor=MUTED))
        return d

    ev = section("Visual Evidence") + [imgs]
    if record.gradcam_filename:
        ev += [Spacer(1, 3), legend_drawing(),
               Paragraph("The saliency map is a guide to the model's attention. It does not outline the lesion.", cap)]
    story += [KeepTogether(ev)]

    # ---- 7. Recommended actions ---------------------------------------------
    nums = [Paragraph(f"<b><font color='#5b21b6'>{i}.</font></b>  " + esc(r), bul)
            for i, r in enumerate(summary["recommendations"], 1)]
    story += [KeepTogether(section("Recommended Actions") + nums +
              [Paragraph("General guidance for the treating clinician to confirm; it does not replace "
                         "individual clinical judgement.", small)])]

    # ---- 8. Limitations ------------------------------------------------------
    lim = [
        "Results depend on image quality, framing, focus and lighting, and may be less reliable for images "
        "that differ from the model's training data.",
        "The saliency map is indicative only and is not a lesion segmentation or measurement.",
        "This tool has not been validated as a stand-alone diagnostic device.",
    ]
    story += [KeepTogether(section("Limitations") + [Paragraph("\u2022  " + esc(t), sbul) for t in lim])]

    # ---- 9. Sign-off + disclaimer -----------------------------------------------
    sign = Table([["", "", ""],
                  [Paragraph("Reviewing clinician (name and signature)", small),
                   Paragraph("Registration no.", small), Paragraph("Date and stamp", small)]],
                 colWidths=[CW * 0.44, CW * 0.26, CW * 0.30], rowHeights=[11 * mm, None])
    sign.setStyle(TableStyle([
        ("LINEABOVE", (0, 1), (-1, 1), 0.6, TEXT),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-2, -1), 14),
    ]))
    disclaimer = Table([[Paragraph(
        "<b>Disclaimer:</b> This AI-generated analysis is intended for clinical decision support only and does "
        "not constitute a final medical diagnosis. All results must be reviewed by a qualified healthcare "
        "professional before any treatment planning.", disc)]], colWidths=[CW])
    disclaimer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#fecaca")),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]))
    story += [Spacer(1, 8), KeepTogether([sign, Spacer(1, 8), disclaimer])]

    doc = BaseDocTemplate(filepath, pagesize=A4, leftMargin=LM, rightMargin=RM,
                          topMargin=32 * mm, bottomMargin=20 * mm,
                          title=f"DiaVision AI Screening Report {report_no}", author="DiaVision AI")
    frame = Frame(LM, 20 * mm, CW, H - 32 * mm - 20 * mm, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    doc.build(story)'''


def find_func(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node.lineno, node.end_lineno
    return None


def main():
    if not os.path.exists(TARGET):
        sys.exit(f"Cannot find {TARGET}. Run this from your project folder (next to app.py).")
    src = open(TARGET, encoding="utf-8").read()
    if MARK in src:
        sys.exit("Already patched. Nothing to do.")

    tree = ast.parse(src)
    missing = [n for n in ("build_report_pdf", "clinical_summary", "_jet_colormap", "verdict_for")
               if find_func(tree, n) is None]
    if missing:
        sys.exit("This app.py is missing: " + ", ".join(missing) +
                 "\nIt looks different from what this patch expects, so nothing was changed.")

    lines = src.split("\n")
    s, e = find_func(tree, "build_report_pdf")
    lines[s - 1:e] = NEW_REPORT.strip("\n").split("\n")
    new_src = "\n".join(lines)

    try:
        compile(new_src, TARGET, "exec")
    except SyntaxError as exc:
        sys.exit(f"Patch would create a syntax error ({exc}). Nothing was changed.")

    backup = TARGET + ".bak"
    n = 1
    while os.path.exists(backup):
        n += 1
        backup = f"{TARGET}.bak{n}"
    shutil.copy(TARGET, backup)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(new_src)
    print(f"Done. Report code replaced. Backup saved as {backup}")
    print("Now restart the app:  python app.py")


if __name__ == "__main__":
    main()