import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


model = load_model('plant_classifier_model.h5')  

datasetpath="structured_dataset" #path of dataset
class_labels = sorted(os.listdir(datasetpath)) 


def predict_image(img_path, class_labels, threshold=0.60, target_size=(224, 224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
    except Exception as e:
        print(" Error loading image:", e)
        return

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    max_conf = np.max(prediction)
    pred_class = np.argmax(prediction)

    if max_conf < threshold:
        print(f"\nPredicted: Unknown (Confidence: {max_conf:.2f})")
    else:
        print(f"\n Predicted Class: {class_labels[pred_class]} (Confidence: {max_conf:.2f})")


while True:
    img_path = input("\nEnter image path (or type 'exit' to quit): ")
    if img_path.lower() == 'exit':
        break
    elif not os.path.exists(img_path):
        print(" File not found. Try again.")
    else:
        predict_image(img_path, class_labels)
