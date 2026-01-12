import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Multi-Disease Chest X-Ray Classification",
    page_icon="🩻",
    layout="centered"
)

# -------------------- TITLE --------------------
st.title("Multi-Disease Chest X-Ray Classification")
st.write("Upload a chest X-ray image to predict the disease")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("best_model.keras")
        return model
    except Exception as e:
        st.error("❌ Failed to load the model.")
        st.exception(e)
        st.stop()

with st.spinner("Loading model... please wait"):
    model = load_model()

st.success("Model loaded successfully")

# -------------------- CLASS NAMES --------------------
class_names = ["COVID", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

# -------------------- FILE UPLOADER --------------------
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

# -------------------- PREDICTION --------------------
if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", width=350)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.write("🧠 Predicting...")

    # Predict
    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions) * 100)

    # -------------------- RESULTS --------------------
    st.subheader("Prediction Result")

    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i] * 100:.2f}%")

    st.markdown("---")
    st.write(f"**Final Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    if confidence < 50:
        st.warning(
            "⚠️ Low confidence prediction. "
            "This may be due to unclear X-ray features or dataset imbalance."
        )
