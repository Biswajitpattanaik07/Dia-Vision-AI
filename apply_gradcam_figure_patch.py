"""
apply_gradcam_figure_patch.py - makes the Grad-CAM output look like a published figure:
(a) Image   (b) Grad-CAM heatmap   (c) Superimposed visualization, with a colour scale.

It changes ONLY the Grad-CAM image code, the result route, and (if you applied the new report
layout) the Grad-CAM part of the PDF inside your own app.py. A backup is saved first.

Run from the project folder:
    python apply_gradcam_figure_patch.py
Then replace templates/result.html with the new result.html and restart:  python app.py
Run a NEW analysis to see the new figure (older results keep their old look).

Optional .env setting:  GRADCAM_STYLE=classic   (default, full-frame like published figures)
                        GRADCAM_STYLE=focus     (only the strongest regions are coloured)
"""
import ast
import os
import shutil
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "app.py"
MARK_A = "# --- grad-cam figure v1 ---"
MARK_B = "# --- grad-cam figure pdf v1 ---"

NEW_GEN = r'''# --- grad-cam figure v1 ---
def _rainbow_colormap(v):
    """Values 0-1 -> uint8 RGB, blue -> cyan -> green -> yellow -> red. The same
    scale is drawn as the colour bar on the result page and in the PDF."""
    import numpy as np
    v = np.clip(np.asarray(v, dtype="float32"), 0, 1)
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    r = np.interp(v, xs, [0, 0, 0, 255, 255])
    g = np.interp(v, xs, [0, 255, 255, 255, 0])
    b = np.interp(v, xs, [255, 255, 0, 0, 0])
    return np.stack([r, g, b], axis=-1).astype("uint8")


def _render_heatmap_only(original, heatmap):
    """Panel (b): the smooth, full-frame Grad-CAM map with no threshold or mask.
    Values are normalised to 0-1 for this image."""
    import numpy as np
    from PIL import Image, ImageFilter
    h = np.clip(np.asarray(heatmap, dtype="float32"), 0, 1)
    img = Image.fromarray(np.uint8(255 * h)).resize(original.size, resample=Image.BICUBIC)
    cell = min(original.size) / max(h.shape)                 # pixels per heatmap cell
    img = img.filter(ImageFilter.GaussianBlur(max(2, int(cell * 0.3))))
    heat = np.array(img).astype("float32") / 255.0
    if heat.max() > 0:
        heat = heat / heat.max()
    return _rainbow_colormap(heat)


def _render_superimposed(original, heat_rgb):
    """Panel (c): the photograph blended with the heatmap, as in published figures."""
    import numpy as np
    base = np.array(original.convert("RGB")).astype("float32")
    return np.clip(0.55 * base + 0.45 * heat_rgb.astype("float32"), 0, 255).astype("uint8")


def generate_gradcam(image_path, class_idx):
    """Builds the Grad-CAM figure images for the given image and predicted class
    index and returns the filename of the superimposed image (gradcam_<name>.png).
    A second file, heat_<name>.png, holds the heatmap-only panel. Raises
    RuntimeError with a clear message on failure (callers catch it and show it).

    GRADCAM_METHOD in .env:  gradcam (default, fast) or occlusion (slower).
    GRADCAM_STYLE  in .env:  classic (default, full-frame like published figures)
                             or focus (only the strongest regions are coloured)."""
    if MODEL is None:
        raise RuntimeError("Model is not loaded.")

    import numpy as np
    import tensorflow as tf
    from PIL import Image, ImageFilter

    method = os.environ.get("GRADCAM_METHOD", "gradcam").strip().lower()
    style = os.environ.get("GRADCAM_STYLE", "classic").strip().lower()
    target_size = MODEL.input_shape[1:3]
    original = Image.open(image_path).convert("RGB")
    arr_np = preprocess_pil(original, target_size)

    try:
        if method == "gradcam":
            heatmap = _gradient_heatmap(arr_np, class_idx)
        else:
            heatmap = _occlusion_heatmap(arr_np, class_idx)
    except Exception as exc:
        raise RuntimeError(f"[v7:{method}] {exc}")

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heat_rgb = _render_heatmap_only(original, heatmap)
    if style == "focus":
        overlay = _render_overlay(original, heatmap)
    else:
        overlay = _render_superimposed(original, heat_rgb)

    base = os.path.splitext(os.path.basename(image_path))[0]
    gradcam_filename = f"gradcam_{base}.png"
    Image.fromarray(overlay).save(os.path.join(UPLOAD_FOLDER, gradcam_filename))
    Image.fromarray(heat_rgb).save(os.path.join(UPLOAD_FOLDER, f"heat_{base}.png"))
    return gradcam_filename'''

NEW_RESULT = r'''def result(analysis_id):
    record = Analysis.query.get_or_404(analysis_id)
    sorted_probs = sorted(record.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    # Newer analyses also have a heatmap-only image (heat_<name>.png) for the 3-panel Grad-CAM figure.
    heat_filename = None
    gname = record.gradcam_filename or ""
    if gname.startswith("gradcam_"):
        candidate = "heat_" + gname[len("gradcam_"):]
        if os.path.exists(os.path.join(UPLOAD_FOLDER, candidate)):
            heat_filename = candidate
    return render_template("result.html", record=record, patient=record.patient, sorted_probs=sorted_probs,
                           verdict=verdict_for(record.prediction_label),
                           summary=clinical_summary(record), heat_filename=heat_filename)'''

PDF_START = "    img_path = os.path.join(UPLOAD_FOLDER, record.image_filename)\n    left = fit_img(img_path, 72 * mm, 54 * mm)"
PDF_END = "    story += [KeepTogether(ev)]\n"
NEW_PDF = r'''    # --- grad-cam figure pdf v1 ---
    img_path = os.path.join(UPLOAD_FOLDER, record.image_filename)
    gname = record.gradcam_filename or ""
    gp = os.path.join(UPLOAD_FOLDER, gname) if gname else None
    hp = os.path.join(UPLOAD_FOLDER, "heat_" + gname[len("gradcam_"):]) if gname.startswith("gradcam_") else None
    have_overlay = bool(gp and os.path.exists(gp))
    have_heat = bool(hp and os.path.exists(hp))

    def legend_drawing(cmap):
        d = Drawing(CW, 11 * mm)
        n, bw = 60, 56 * mm
        x0 = (CW - bw) / 2
        cols = cmap(np.linspace(0, 1, n)[None, :])[0]
        for i in range(n):
            d.add(Rect(x0 + i * bw / n, 5 * mm, bw / n + 0.3, 2.6 * mm, strokeColor=None,
                       fillColor=colors.Color(cols[i][0] / 255, cols[i][1] / 255, cols[i][2] / 255)))
        for frac, txt in ((0, "0.0"), (0.5, "0.5"), (1, "1.0")):
            d.add(String(x0 + bw * frac, 1.2 * mm, txt, textAnchor="middle", fontName="Helvetica",
                         fontSize=6.5, fillColor=MUTED))
        d.add(String(x0 - 2 * mm, 5.6 * mm, "Low influence", textAnchor="end", fontName="Helvetica",
                     fontSize=6.8, fillColor=MUTED))
        d.add(String(x0 + bw + 2 * mm, 5.6 * mm, "High influence", fontName="Helvetica",
                     fontSize=6.8, fillColor=MUTED))
        return d

    if have_overlay and have_heat:
        pw_ = CW / 3
        cells = [fit_img(img_path, pw_ - 6 * mm, 50 * mm) if os.path.exists(img_path)
                 else Paragraph("Image not available.", cap),
                 fit_img(hp, pw_ - 6 * mm, 50 * mm), fit_img(gp, pw_ - 6 * mm, 50 * mm)]
        imgs = Table([cells, [Paragraph("(a) Image", cap), Paragraph("(b) Grad-CAM heatmap", cap),
                              Paragraph("(c) Superimposed visualization", cap)]], colWidths=[pw_] * 3)
        imgs.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
            ("BOX", (0, 0), (0, -1), 0.6, BORDER), ("BOX", (1, 0), (1, -1), 0.6, BORDER),
            ("BOX", (2, 0), (2, -1), 0.6, BORDER),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fafafa")),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        ev = section("Visual Evidence (Grad-CAM)") + [imgs, Spacer(1, 3), legend_drawing(_rainbow_colormap),
             Paragraph("Grad-CAM for the predicted class (" + esc(record.prediction_label) + "). Colour scale: "
                       "relative influence, normalised for each image (0 = lowest, 1 = highest). The map guides "
                       "attention and does not outline the lesion.", cap)]
    else:
        left = fit_img(img_path, 72 * mm, 54 * mm) if os.path.exists(img_path) else Paragraph("Image not available.", cap)
        right = Paragraph("Saliency map not available for this analysis.", cap)
        if have_overlay:
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
        ev = section("Visual Evidence") + [imgs]
        if have_overlay:
            ev += [Spacer(1, 3), legend_drawing(_jet_colormap),
                   Paragraph("The saliency map is a guide to the model's attention. It does not outline the lesion.", cap)]
    story += [KeepTogether(ev)]'''


def find_func(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node.lineno, node.end_lineno
    return None


def main():
    if not os.path.exists(TARGET):
        sys.exit(f"Cannot find {TARGET}. Run this from your project folder (next to app.py).")
    src = open(TARGET, encoding="utf-8").read()
    applied, skipped = [], []

    # ---- part A: figure images + result route ---------------------------------
    if MARK_A not in src:
        tree = ast.parse(src)
        missing = [n for n in ("generate_gradcam", "result", "_gradient_heatmap", "_occlusion_heatmap",
                               "_render_overlay", "_jet_colormap", "preprocess_pil")
                   if find_func(tree, n) is None]
        if missing:
            sys.exit("This app.py is missing: " + ", ".join(missing) +
                     "\nIt looks different from what this patch expects, so nothing was changed.")
        lines = src.split("\n")
        spans = sorted([(find_func(tree, "result"), NEW_RESULT), (find_func(tree, "generate_gradcam"), NEW_GEN)],
                       key=lambda t: t[0][0], reverse=True)          # later function first
        for (s, e), text in spans:
            lines[s - 1:e] = text.strip("\n").split("\n")
        src = "\n".join(lines)
        applied.append("Grad-CAM figure images and result route")
    else:
        skipped.append("figure images and result route (already applied)")

    # ---- part B: the PDF (only if the new report layout is installed) ----------
    if MARK_B not in src:
        a = src.find(PDF_START)
        b = src.find(PDF_END, a) if a != -1 else -1
        if a != -1 and b != -1:
            src = src[:a] + NEW_PDF.strip("\n") + "\n" + src[b + len(PDF_END):]
            applied.append("PDF report Grad-CAM figure")
        else:
            skipped.append("PDF report (the new report layout is not installed; run apply_report_patch.py "
                           "first, then run this script again)")
    else:
        skipped.append("PDF report (already applied)")

    if not applied:
        sys.exit("Nothing to do. " + "; ".join(skipped))

    try:
        compile(src, TARGET, "exec")
    except SyntaxError as exc:
        sys.exit(f"Patch would create a syntax error ({exc}). Nothing was changed.")

    backup = TARGET + ".bak"
    n = 1
    while os.path.exists(backup):
        n += 1
        backup = f"{TARGET}.bak{n}"
    shutil.copy(TARGET, backup)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)
    print("Applied: " + "; ".join(applied))
    if skipped:
        print("Skipped: " + "; ".join(skipped))
    print(f"Backup saved as {backup}")
    print("Next: replace templates/result.html with the new one, then restart:  python app.py")


if __name__ == "__main__":
    main()