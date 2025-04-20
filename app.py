import streamlit as st
st.cache_data.clear()  # Clear cached data if any

from PIL import Image
import numpy as np
from main import process_blood_smear

st.set_page_config(page_title="Leukemia Classifier", layout="centered")

st.title("Leukemia Classification")

st.write("Upload a blood smear image to classify it for potential signs of leukemia.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image") # use_container_width=True

    # Save the uploaded image to a temporary file
    image_path = "temp_input_image.png"
    image.save(image_path)

    # Button to trigger the classification
    if st.button("Classify Image"):
        with st.spinner("Analyzing image..."):
            try:
                # Call process_blood_smear from main.py
                result = process_blood_smear(image_path)
                st.success(f"Prediction: **{result}**")
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")