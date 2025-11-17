import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import io
from PIL import Image
import cv2

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
CLASS_NAMES = ["No Tumor", "Tumor"]  
TARGET_SIZE = (128, 128)             
RESCALE = 1.0 / 255.0                


@st.cache_resource
def load_model(model_path: str):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

def ensure_3_channels(img: Image.Image) -> Image.Image:
    # Convert to RGB
    return img.convert("RGB")

def preprocess_image(pil_image: Image.Image, target_size=TARGET_SIZE) -> np.ndarray:
    pil_image = ensure_3_channels(pil_image)
    pil_image = pil_image.resize(target_size, Image.BILINEAR)
    arr = np.array(pil_image).astype("float32")
    arr *= RESCALE
    return np.expand_dims(arr, axis=0)

def to_class_probs(pred: np.ndarray) -> np.ndarray:
    pred = np.array(pred)
    if pred.ndim == 2 and pred.shape[1] == 1:
        p1 = float(pred[0, 0])
        return np.array([1 - p1, p1], dtype="float32")
    if pred.ndim == 2 and pred.shape[1] == 2:
        return pred[0].astype("float32")
    e = np.exp(pred[0] - np.max(pred[0]))
    return (e / e.sum()).astype("float32")

def find_last_conv_layer(model: tf.keras.Model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def grad_cam(model: tf.keras.Model, img_batch: np.ndarray, last_conv_name: str | None = None, class_index: int | None = None):
    if last_conv_name is None:
        last_conv_name = find_last_conv_layer(model)
    if last_conv_name is None:
        raise ValueError("No Conv2D layer found for Grad-CAM.")

    last_conv_layer = model.get_layer(last_conv_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_out)  
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  

    conv_out = conv_out[0] 
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1)

    # Normalize to [0,1]
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap_on_image(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (pil_image.width, pil_image.height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    base = np.array(pil_image.convert("RGB"))
    overlay = cv2.addWeighted(base, 1.0, heatmap_color, alpha, 0)
    return Image.fromarray(overlay)


model = load_model("brain_tumor_classifier_cnn.h5")

st.title("🧠 Brain Tumor Classifier (Demo)")
st.markdown("Upload a brain MRI image and get a model prediction with confidence. "
            "**Not for medical use.**")

with st.sidebar:
    st.header("Options")
    show_explain = st.checkbox("Show Grad-CAM heatmap (explainability)", value=True)
    st.caption("Grad-CAM highlights image regions that influenced the prediction.")
    st.divider()
    st.caption("Tip: Use images similar to your training set (orientation, contrast, etc.).")

uploaded_file = st.file_uploader("Choose an MRI image…", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Read & show image
        pil_img = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

        # Preprocess & predict
        batch = preprocess_image(pil_img, target_size=TARGET_SIZE)
        raw_pred = model.predict(batch, verbose=0)
        probs = to_class_probs(raw_pred)           # convert to class probabilities
        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]
        conf_pct = float(probs[pred_idx] * 100)

        # Results
        st.subheader(f"Prediction: **{pred_label}**")
        st.write(f"Confidence: **{conf_pct:.2f}%**")

        # Probability chart
        df_probs = pd.DataFrame(
            {"Class": CLASS_NAMES, "Probability": probs}
        ).set_index("Class")
        st.bar_chart(df_probs)

        # Explainability
        if show_explain:
            try:
                heatmap = grad_cam(model, batch, class_index=pred_idx)
                overlay = overlay_heatmap_on_image(pil_img, heatmap, alpha=0.35)
                st.image(overlay, caption="Grad-CAM heatmap overlay", use_container_width=True)
            except Exception as e:
                st.info(f"Grad-CAM not available: {e}")

        # Clinical disclaimer
        st.divider()
        st.caption("⚠️ This is a demo research tool. Do not use for diagnosis or treatment decisions.")

    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Upload a JPG or PNG MRI image to begin.")
