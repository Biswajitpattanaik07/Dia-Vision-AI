
"""
apply_heatmap_patch.py - makes the heatmap MUCH faster and less speckled.

It changes ONLY the heatmap functions inside your own app.py (everything else,
such as your Developer page and chat format, is left exactly as it is).
A backup is saved first as app.py.bak.

Run from the project folder:
    python apply_heatmap_patch.py
Then restart the app:  python app.py
"""
import ast
import os
import shutil
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "app.py"
MARK = "# --- fast heatmap patch v20 ---"

NEW_GRADIENT = '''# --- fast heatmap patch v20 ---
_CAM_ENGINES = {}


def _get_cam_engine(layer_names, plus):
    """Builds ONCE (then caches) a compiled function that returns the feature maps
    and gradients of several layers in a single forward+backward pass. Rebuilding
    the sub-model on every request was the main cause of slow analyses."""
    import tensorflow as tf
    key = (id(MODEL), tuple(layer_names), plus)
    if key in _CAM_ENGINES:
        return _CAM_ENGINES[key]

    head = MODEL.layers[-1]
    act_name = getattr(getattr(head, "activation", None), "__name__", "")
    use_logits = hasattr(head, "kernel") and act_name in ("softmax", "sigmoid")
    second = head.input if use_logits else MODEL.output
    layers = [MODEL.get_layer(n) for n in layer_names]
    gm = tf.keras.models.Model(MODEL.inputs, [l.output for l in layers] + [second])
    n_layers = len(layers)

    @tf.function(reduce_retracing=True)
    def step(batch, class_idx):
        with tf.GradientTape() as tape:
            outs = gm(batch, training=False)
            convs = list(outs[:n_layers])
            out = outs[n_layers]
            if isinstance(out, (list, tuple)):
                out = out[0]
            if use_logits:  # raw score: softmax saturates and kills gradients
                out = tf.matmul(out, head.kernel)
                if getattr(head, "use_bias", False):
                    out = out + head.bias
            score = tf.reduce_sum(tf.gather(out, class_idx, axis=1))
        grads = tape.gradient(score, convs)
        return convs, grads

    _CAM_ENGINES[key] = step
    return step


def _cams_multi(arr_np, class_idx, layer_names, plus=False, tta=3):
    """CAMs for several layers from ONE batched pass (original + flipped copies,
    flips undone and averaged). Returns one 2D map (0-1) per layer."""
    import numpy as np
    import tensorflow as tf

    step = _get_cam_engine(layer_names, plus)
    a = arr_np[0]
    flips = [None, 1, 0][:max(1, min(3, tta))]          # none, horizontal, vertical
    batch = np.stack([a if f is None else np.ascontiguousarray(np.flip(a, axis=f)) for f in flips])
    convs, grads = step(tf.constant(batch), tf.constant(int(class_idx), dtype=tf.int32))

    results = []
    for conv_t, grad_t in zip(convs, grads):
        if grad_t is None:
            raise RuntimeError("no gradients for a heatmap layer")
        conv, g = conv_t.numpy(), grad_t.numpy()
        cams = []
        for i, f in enumerate(flips):
            c, gr = conv[i], g[i]
            if plus:  # Grad-CAM++ weighting
                g2 = gr ** 2
                g3 = g2 * gr
                denom = 2 * g2 + c.sum(axis=(0, 1), keepdims=True) * g3
                denom = np.where(denom != 0, denom, 1e-8)
                weights = ((g2 / denom) * np.maximum(gr, 0)).sum(axis=(0, 1))
            else:
                weights = gr.mean(axis=(0, 1))
            cam = np.maximum((c * weights).sum(axis=-1), 0)
            if f is not None:
                cam = np.flip(cam, axis=f)               # undo the flip on the CAM
            cams.append(cam / (cam.max() + 1e-8))
        cam = np.mean(cams, axis=0)
        results.append(cam / (cam.max() + 1e-8))
    return results


def _gradient_heatmap(arr_np, class_idx):
    """Fast, sharper Grad-CAM: one batched pass gives both the coarse (last layer)
    and fine (earlier layer) maps, which are fused with a geometric mean.
    Falls back to the plain method on any problem. arr_np: numpy (1,H,W,3).
    Settings (.env): GRADCAM_FUSE=1/0, GRADCAM_TTA=1..3, GRADCAM_PLUS=0/1,
    GRADCAM_EDGE_TRIM, GRADCAM_LAYER, GRADCAM_FINE_LAYER."""
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    try:
        plus = os.environ.get("GRADCAM_PLUS", "0").strip() == "1"
        tta = int(os.environ.get("GRADCAM_TTA", "3"))
        trim = int(os.environ.get("GRADCAM_EDGE_TRIM", "1"))
        coarse_layer = _pick_gradcam_layer(MODEL)
        fuse = os.environ.get("GRADCAM_FUSE", "1").strip() != "0"
        fine_layer = _pick_fine_layer(coarse_layer) if fuse else None
        names = [coarse_layer.name] + ([fine_layer.name] if fine_layer is not None else [])

        cams = _cams_multi(arr_np, class_idx, names, plus, tta)

        def _trim(cam):
            if trim > 0 and min(cam.shape) > 2 * trim + 2:
                cam = cam.copy()
                cam[:trim, :] = 0; cam[-trim:, :] = 0
                cam[:, :trim] = 0; cam[:, -trim:] = 0
                cam = cam / (cam.max() + 1e-8)
            return cam

        coarse = _trim(cams[0])
        if len(cams) == 1:
            print(f"[DiaVision] Heatmap: layer {coarse_layer.name} (no fusion), {tta}x flip-averaged")
            return coarse

        fine = _trim(cams[1])
        coarse_up = np.array(Image.fromarray(np.uint8(255 * coarse)).resize(
            (fine.shape[1], fine.shape[0]), Image.BICUBIC)).astype("float32") / 255.0
        fused = np.sqrt(np.clip(coarse_up, 0, 1) * np.clip(fine, 0, 1))
        print(f"[DiaVision] Heatmap: fused {coarse_layer.name} + {fine_layer.name}"
              f"{' (Grad-CAM++)' if plus else ''}, {tta}x flip-averaged (single pass)")
        return fused / (fused.max() + 1e-8)
    except Exception as exc:
        print(f"[DiaVision] Fast heatmap failed ({exc}); using the plain method.")
        return _gradient_heatmap_legacy(tf.convert_to_tensor(arr_np), class_idx)
'''

NEW_RENDER = '''def _render_overlay(original, heatmap):
    """Blends a small 2D heatmap (values 0-1) onto the original PIL image.
    Smoothing now scales with the heatmap's cell size, so a coarse 12x12 map
    becomes soft connected regions instead of scattered dots."""
    import numpy as np
    from PIL import Image, ImageFilter

    # 1) upsample to the original size, then smooth (radius follows the cell size)
    heat_img = Image.fromarray(np.uint8(255 * np.clip(heatmap, 0, 1))).resize(original.size, resample=Image.BICUBIC)
    cell = min(original.size) / max(np.asarray(heatmap).shape)      # pixels per heatmap cell
    radius = max(min(original.size) // 45, int(cell * 0.45), 2)
    heat_img = heat_img.filter(ImageFilter.GaussianBlur(radius))
    heat = np.array(heat_img).astype("float32") / 255.0

    # 2) keep the heatmap on the subject (ignore pure-black background)
    gray = np.array(original).astype("float32").mean(axis=2)
    mask_img = Image.fromarray(np.uint8((gray > 20) * 255)).filter(ImageFilter.MaxFilter(5))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(3))
    heat = heat * (np.array(mask_img).astype("float32") / 255.0)

    # 3) normalise using the 99.5th percentile, then drop weak areas
    hi = np.percentile(heat, 99.5)
    if hi > 0:
        heat = np.clip(heat / hi, 0, 1)
    cutoff = float(os.environ.get("GRADCAM_THRESHOLD", "0.35"))
    heat = np.clip((heat - cutoff) / (1 - cutoff), 0, 1)

    # 4) blend: strong areas get colour, everything else stays the original photo
    colored = _jet_colormap(heat).astype("float32")
    original_arr = np.array(original).astype("float32")
    alpha = (0.7 * heat)[..., None]
    return np.clip(original_arr * (1 - alpha) + colored * alpha, 0, 255).astype("uint8")
'''


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
    needed = ["_gradient_heatmap", "_render_overlay", "_gradient_heatmap_legacy",
              "_pick_gradcam_layer", "_pick_fine_layer", "_jet_colormap"]
    missing = [n for n in needed if find_func(tree, n) is None]
    if missing:
        sys.exit("This app.py is missing: " + ", ".join(missing) +
                 "\nIt looks different from what this patch expects, so nothing was changed.")

    lines = src.split("\n")
    # replace the later function first so earlier line numbers stay valid
    s, e = find_func(tree, "_render_overlay")
    lines[s - 1:e] = NEW_RENDER.rstrip("\n").split("\n")
    s, e = find_func(tree, "_gradient_heatmap")
    lines[s - 1:e] = NEW_GRADIENT.rstrip("\n").split("\n")
    new_src = "\n".join(lines)
    new_src = new_src.replace("Explanation code version: v18", "Explanation code version: v18 + fast heatmap")

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
    print(f"Done. Heatmap code replaced. Backup saved as {backup}")
    print("Now restart the app:  python app.py")


if __name__ == "__main__":
    main()