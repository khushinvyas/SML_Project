import streamlit as st
import numpy as np
import os
import pydicom
from skimage.transform import resize
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model
import joblib  # For loading the trained SVM model
import time  # For simulating processing delay
import base64


# Function definitions for preprocessing and feature extraction
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

# Load the SVM model
svm_model_path = r'svm_model2.pkl'
if os.path.exists(svm_model_path):
    svm_model = joblib.load(svm_model_path)
else:
    st.error('Model file not found. Please check the model path.')
    st.stop()



# Streamlit app layout
st.title('Medical Image Tampering Detection')
uploaded_file = st.file_uploader("Choose a DICOM file", type="dcm")

if uploaded_file is not None:
    with st.spinner('Processing...'):
        preprocessed_img, original_img = load_and_preprocess_dicom(uploaded_file)
        features = extract_features(preprocessed_img)
        prediction = svm_model.predict([features])[0]

        st.success('Processing complete!')

       # Display the uploaded DICOM image
        st.image(original_img, caption='Uploaded DICOM Image', width=800, clamp=True)

        # Add a dictionary to map the prediction to the description
        prediction_dict = {
            'TB': 'True-Benign: A location that actually has no cancer',
            'TM': 'True-Malicious: A location that has real cancer',
            'FB': 'False-Benign: A location that has real cancer, but it was removed',
            'FM': 'False-Malicious: A location that does not have cancer, but fake cancer was injected there'
        }
        
        # Use the dictionary to get the description for the prediction
        prediction_description = prediction_dict.get(prediction, 'Unknown prediction')
        # Highlight the prediction output
        if prediction in ['FB', 'FM']:
            st.markdown(f'<p style="color:red">Prediction: {prediction_description}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color:green">Prediction: {prediction_description}</p>', unsafe_allow_html=True)


st.markdown("""
To test the model, upload a DICOM image of a lung scan. The model will classify the image as tampered or untampered.
""")
