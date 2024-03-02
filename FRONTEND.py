import cv2
import numpy as np
from PIL import ImageChops, ImageEnhance, Image
from tensorflow.keras.models import load_model
import streamlit as st
import os

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 91).resize((128, 128))).flatten() / 255.0

def main():
    st.title("Image Authenticity Checker")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Save the uploaded image to a temporary file
        temp_path = 'temp_uploaded_image.jpg'
        uploaded_image = Image.open(uploaded_file)
        uploaded_image.save(temp_path)

        # Load the pre-trained model
        model = load_model(r"D:\PYTHON\TECHSAKSHAM\Project\model_casia_run1.h5")

        # Prepare and predict on the image
        image = prepare_image(temp_path)
        image = image.reshape(-1, 128, 128, 3)
        y_pred = model.predict(image)

        # Display the result
        if y_pred[0][0] > 0.5:
            st.success("Real Image")
        else:
            st.error("Fake Image")

        # Remove the temporary file
        os.remove(temp_path)

if __name__ == "__main__":
    main()
