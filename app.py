import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# ==========================
# Load Model
# ==========================
MODEL_PATH = "waste_resnet50_final.keras"   # model path
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Class Names (same order as training)
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

st.title("♻️ Waste Classification Web App")
st.write("Upload an image and the model will predict its category.")

# ==========================
# File Upload
# ==========================
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ==========================
    # Preprocess Image
    # ==========================
    img = img.resize((224, 224))   # match training input
    img_array = np.array(img) / 255.0   #  normalize
    img_array = np.expand_dims(img_array, axis=0)

    # ==========================
    # Prediction
    # ==========================
    preds = model.predict(img_array)
    confidence = np.max(preds)
    class_idx = np.argmax(preds)

    # ✅ Show Debug Info
    st.write("Class Order:", class_names)
    st.write("Raw Prediction:", preds)

    if confidence < 0.6:
        st.warning(f" Not confident. Maybe: **{class_names[class_idx]}** ({confidence:.2f})")
    else:
        st.success(f" Predicted: **{class_names[class_idx]}** | Confidence: {confidence:.2f}")

