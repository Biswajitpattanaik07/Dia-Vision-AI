import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="DiaVision AI")
st.title("DiaVision AI")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/best_model.keras")

model = load_model()
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)
    h, w = model.input_shape[1:3]
    arr = np.expand_dims(np.array(img.resize((w, h))).astype("float32"), 0)
    preds = model.predict(arr, verbose=0)[0]
    st.success(f"{CLASS_NAMES[int(np.argmax(preds))]} ({preds.max()*100:.1f}%)")
    st.bar_chart({n: float(p) for n, p in zip(CLASS_NAMES, preds)})
