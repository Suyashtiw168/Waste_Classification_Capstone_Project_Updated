import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown

# ✅ File ID from Google Drive
FILE_ID = "1ZkWHE34de9bH2dWgVa3PJkEwchtLlsVX"
MODEL_FILE = "waste_resnet50_final.keras"

# ✅ Download model if not present
if not os.path.exists(MODEL_FILE):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)

# ✅ Load model
model = tf.keras.models.load_model(MODEL_FILE)

class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# === Rest of Streamlit UI ===
st.title("♻️ Waste Classification App")
uploaded_file = st.file_uploader("Upload image...", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224,224))   # ye size match karo training size
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    label = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    st.success(f"✅ Predicted: {label} (Confidence: {confidence:.2f})")
