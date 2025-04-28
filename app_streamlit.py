import streamlit as st
import tensorflow as tf
import numpy as np
from utils import preprocess_image
from PIL import Image

# Load model
model = tf.keras.models.load_model('cnn_cifar10_model.h5')

# Class names CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("🚀 Image Recognition Demo")
st.write("Upload an image and let the model predict what it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(uploaded_file)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### 🎯 Prediction: **{predicted_class}**")
