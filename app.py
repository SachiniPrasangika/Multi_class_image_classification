import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Page configuration
st.set_page_config(page_title="AI Image Classifier", page_icon="🤖")

st.title("AI Image Classification App")
st.write("Upload an image to classify it using the trained MobileNetV2 model.")

# Load model and class names
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('trained_model.keras')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

try:
    model, class_names = load_resources()
except Exception as e:
    st.error(f"Error loading model or class_names.json: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    if st.button("Predict"):
        predictions = model.predict(img_array)
        class_idx = str(np.argmax(predictions))
        predicted_class = class_names[class_idx]
        confidence = 100 * np.max(predictions)
        
        # Display results
        st.subheader(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")