import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

st.title("AI Image Classification App")

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('trained_model.keras')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_resources()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB') # Convert to RGB to ensure 3 channels
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("Predict"):
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0]) # Get index of max probability
        predicted_class = class_names[str(class_idx)]
        confidence = 100 * np.max(predictions[0])
        
        st.subheader(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.write("Raw probabilities:", predictions[0]) # Mehen balanna puluwan wena class walata probability thiyenawada kiyala
