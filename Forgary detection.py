import cv2
import numpy as np
from PIL import ImageChops, ImageEnhance, Image
from tensorflow.keras.models import load_model

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

img_path = r"C:\Users\abhis\OneDrive\Desktop\WhatsApp Image 2024-02-26 at 5.04.06 PM.jpeg"
img = cv2.imread(img_path)
model = load_model(r"D:\PYTHON\TECHSAKSHAM\Project\model_casia_run1.h5")

image = prepare_image(img_path)
image = image.reshape(-1, 128, 128, 3)
y_pred = model.predict(image)

if y_pred[0][0] > 0.5:
    print("real")
else:
    print("fake")
