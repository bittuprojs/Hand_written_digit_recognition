import streamlit as st
import numpy as np
from PIL import Image
from OpenCV import cv2
import os
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Digit Recognition", page_icon="✍️", layout="centered")

st.title("✍️ Handwritten Digit Recognition (MNIST · CNN)")
st.caption("Draw a digit (0–9) below. The app preprocesses your drawing to MNIST format (28×28, grayscale) and predicts with a CNN.")

@st.cache_resource
def get_model():
    path = "mnist_cnn_model.h5"
    if not os.path.exists(path):
        st.error("Model file 'mnist_cnn_model.h5' not found. Run train_model.py locally to create it, then restart the app.")
        st.stop()
    return load_model(path)

model = get_model()

# Sidebar controls
st.sidebar.header("Canvas Settings")
stroke_width = st.sidebar.slider("Stroke width", 5, 40, 20)
realtime = st.sidebar.checkbox("Predict in real-time", value=False)
st.sidebar.markdown("---")
st.sidebar.write("Tip: Write centrally with thick strokes for best accuracy.")

CANVAS_SIZE = 280  # Larger canvas that's downscaled to 28×28
canvas_result = st_canvas(
    fill_color="#00000000",
    stroke_width=stroke_width,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=CANVAS_SIZE,
    width=CANVAS_SIZE,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess(image_data: np.ndarray) -> np.ndarray:
    """Convert RGBA canvas image (float 0-255) to MNIST-like 28x28 tensor."""
    if image_data is None:
        return None
    # RGBA -> Grayscale
    img = image_data[:, :, :3].astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Invert so strokes become white on black (MNIST style)
    gray = 255 - gray

    # Normalize lighting and denoise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize with Otsu
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find bounding box of the digit
    ys, xs = np.where(bw > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None  # empty canvas
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    digit = bw[y_min:y_max+1, x_min:x_max+1]

    # Make square by padding
    h, w = digit.shape
    size = max(h, w)
    pad_y = (size - h) // 2
    pad_x = (size - w) // 2
    digit_sq = cv2.copyMakeBorder(digit, pad_y, size - h - pad_y, pad_x, size - w - pad_x, cv2.BORDER_CONSTANT, value=0)

    # Resize to 20x20 then pad to 28x28 (similar to classic MNIST centering)
    digit_20 = cv2.resize(digit_sq, (20, 20), interpolation=cv2.INTER_AREA)
    pad = 4
    digit_28 = np.pad(digit_20, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

    # Normalize to [0,1] and add channel & batch dims
    digit_28 = digit_28.astype("float32") / 255.0
    digit_28 = digit_28.reshape(1, 28, 28, 1)
    return digit_28

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Your Drawing")
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data.astype(np.uint8), caption="Canvas", use_column_width=True)

with col2:
    st.subheader("2) Preprocessed (28×28)")
    processed = preprocess(canvas_result.image_data)
    if processed is not None:
        # Enlarge for display only
        preview = (processed[0, :, :, 0] * 255).astype(np.uint8)
        preview = cv2.resize(preview, (196, 196), interpolation=cv2.INTER_NEAREST)
        st.image(preview, caption="Model Input Preview", clamp=True)
    else:
        st.info("Draw a digit to see the preprocessed view.")

def predict(digit_tensor):
    preds = model.predict(digit_tensor, verbose=0)[0]
    top = int(np.argmax(preds))
    return top, preds

# Predict button (or realtime)
do_predict = False
if st.button("🔮 Predict") or (realtime and processed is not None):
    do_predict = True

if do_predict:
    if processed is None:
        st.warning("Please draw a digit first.")
    else:
        digit, probs = predict(processed)
        st.success(f"Predicted Digit: **{digit}**")
        st.bar_chart(probs)

st.markdown("---")
st.caption("Built with Streamlit, TensorFlow/Keras, and streamlit-drawable-canvas.")
