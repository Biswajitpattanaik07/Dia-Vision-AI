"""
quick_check.py - runs your model on the images in the 'uploads' folder
and prints the raw output numbers, so we can see what the model is doing.

Run from the project folder (venv active):
    python quick_check.py
Optional: python quick_check.py "D:\\some\\other\\folder"
"""
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.path.join("models", "best_model.keras")
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
folder = sys.argv[1] if len(sys.argv) > 1 else "uploads"

model = tf.keras.models.load_model(MODEL_PATH)
size = model.input_shape[1:3]
n_out = model.output_shape[-1]
print(f"\nModel input size: {size}   outputs: {n_out}")

files = [f for f in sorted(os.listdir(folder))
         if f.lower().endswith(EXTS) and not f.startswith("gradcam_")][:25]
if not files:
    print("No images found in:", folder)
    sys.exit()


def load(path, mode):
    im = Image.open(path).convert("RGB").resize((size[1], size[0]))
    a = np.array(im).astype("float32")
    return a / 255.0 if mode == "scale" else a


def fmt(p):
    return "  ".join(f"out{i}={v * 100:5.1f}%" for i, v in enumerate(p))


print("\nImage name".ljust(48), "RAW (0-255)".ljust(40), "SCALED (/255)")
for f in files:
    path = os.path.join(folder, f)
    raw = model.predict(np.expand_dims(load(path, "raw"), 0), verbose=0)[0]
    sca = model.predict(np.expand_dims(load(path, "scale"), 0), verbose=0)[0]
    print(f[:46].ljust(48), fmt(raw).ljust(40), fmt(sca))

# Extra sanity checks with synthetic images
print("\nSanity checks (not real photos):")
for name, arr in (("all black", np.zeros((1, size[0], size[1], 3), "float32")),
                  ("all white", np.full((1, size[0], size[1], 3), 255.0, "float32")),
                  ("random noise", np.random.rand(1, size[0], size[1], 3).astype("float32") * 255)):
    print(name.ljust(48), fmt(model.predict(arr, verbose=0)[0]))