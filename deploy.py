import streamlit as st
import numpy as np
import os
import pandas as pd
import pydicom
from skimage.transform import resize
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model
import joblib  # For loading the trained SVM model
import matplotlib.pyplot as plt
from PIL import Image
import csv

# Preprocess the DICOM image (ensure this matches your earlier preprocessing)
def load_and_preprocess_dicom(file, img_size=224):
    ds = pydicom.dcmread(file)
    img = ds.pixel_array
    if img.ndim == 2:
        img = np.stack((img,) * 3, -1)
    img_resized = resize(img, (img_size, img_size), anti_aliasing=True)
    img_preprocessed = preprocess_input(img_resized)
    return img_preprocessed, img

# Extract features using ResNet50 (updated to match the training process)
def extract_features(img):
    base_model = ResNet50(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    features_flattened = features.reshape(features.shape[0], -1)
    return features_flattened.flatten()

# Function to retrieve tampered coordinates from the CSV files
def get_coordinates(file_paths, target_uuid, target_slice):
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['uuid'] == str(target_uuid) and row['slice'] == str(target_slice):
                        return int(row['x']), int(row['y'])
        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
    # If UUID or slice is not found in any file, return None
    return None, None

# Load the SVM model
svm_model_path = 'svm_model.pkl'
svm_model = None
if os.path.exists(svm_model_path):
    svm_model = joblib.load(svm_model_path)
else:
    st.error('Model file not found. Please check the model path.')
    st.stop()

st.title('Medical Image Tampering Detection')

# User input for UUID
uuid = st.text_input("Enter the UUID (directory name) of the image:")

uploaded_file = st.file_uploader("Choose a DICOM slice file")

file_path1 = "labels_exp1.csv"
file_path2 = "labels_exp2.csv"

if uploaded_file is not None and uuid and svm_model:
    with st.spinner('Processing...'):
        # Assuming the uploaded file name is the slice number
        slice_number = uploaded_file.name.split('.')[0]

        # Attempt to find coordinates in both files
        x_coord, y_coord = get_coordinates([file_path1, file_path2], uuid, slice_number)

        if x_coord is not None and y_coord is not None:
            st.success(f"Coordinates found: x={x_coord}, y={y_coord}")
        else:
            st.error(f"No coordinates found for UUID {uuid} and slice {slice_number}.")
            st.stop()  # Stop execution if no coordinates found

        preprocessed_img, original_img = load_and_preprocess_dicom(uploaded_file)
        features = extract_features(preprocessed_img)
        prediction = svm_model.predict([features])[0]

        fig, ax = plt.subplots()
        ax.imshow(original_img, cmap='gray')

        prediction_dict = {
            'TB': 'True-Benign: A location that actually has no cancer',
            'TM': 'True-Malicious: A location that has real cancer',
            'FB': 'False-Benign: A location that has real cancer, but it was removed',
            'FM': 'False-Malicious: A location that does not have cancer, but fake cancer was injected there'
        }

        prediction_description = prediction_dict.get(prediction, 'Unknown prediction')

        if prediction in ['FB', 'FM']:
            # Highlighting the tampered area with retrieved coordinates
            ax.scatter(x=x_coord, y=y_coord, c='red', s=40)
            st.pyplot(fig)
            st.markdown(f'<p style="color:red">Prediction: {prediction_description}</p>', unsafe_allow_html=True)
        else:
            st.pyplot(fig)
            st.markdown(f'<p style="color:green">Prediction: {prediction_description}</p>', unsafe_allow_html=True)
