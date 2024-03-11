# import cv2
import numpy as np
from PIL import ImageChops, ImageEnhance, Image
from keras.models import load_model
import streamlit as st
import os
import time
import base64

from st_pages import show_pages, Page, add_page_title

st.title("Image Forgery Detector")


def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
         .stApp {{
             
             background-image: url(data:image/png;base64,{b64_encoded});
             background-size:cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    


show_pages(
    [
        Page("FRONTEND.py", "FORGERY DETECTOR", "1️⃣"),
        Page("confidence.py", "CONFIDENCE CALCULATOR", "2️⃣"),

        # Page("image-forgery-detector.ipynb","image-forgery-detector.ipynb", "3️⃣"),
    ]
)
def show_sample_images_sidebar():
    sample_images = ["photo.jpg", "photo1.jpg"]
    
    st.sidebar.subheader("Sample Images for Testing:")
    
    for idx, image_path in enumerate(sample_images):
        st.sidebar.image(image_path, caption=f"Sample Image {idx+1}", use_column_width=True)



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

    set_background("bg3.png")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    show_sample_images_sidebar()
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        progress_bar = st.progress(0)  # Create a progress bar

        for percent_complete in range(0, 101, 10):
            time.sleep(0.2)  # Simulate the classification process, replace with actual code
            progress_bar.progress(percent_complete)

        # Save the uploaded image to a temporary file
        temp_path = 'temp_file_name.jpg'
        uploaded_image = Image.open(uploaded_file)
        uploaded_image.save(temp_path)

        # print("\n\n\n",os.getcwd()+"/model_casia_run1.h5","\n\n\n\n\n\n")
        # Load the pre-trained model
        model = load_model(r"./model_casia_run1.h5")

        # Prepare and predict on the image
        image = prepare_image(temp_path)
        image = image.reshape(-1, 128, 128, 3)
        y_pred = model.predict(image)



        # Display the result
        if y_pred[0][0] > 0.5:
            st.success(f"Real Image")
        else:
            st.error("Tampered image")

        # if y_pred[0][0] > 0.5:
        #     percent = y_pred[0][0] * 100
        #     st.success(f"Real Image: {percent}%")
        # else:
        #     percent = 100 - y_pred[0][0] * 100
        #     if percent > 98.5:
        #         print("image seems to be cropped")
        #     st.error(f"Fake Image: {percent}%")

        # Remove the temporary file
        os.remove(temp_path)


if __name__ == "__main__":
    main()
