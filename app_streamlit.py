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

st.title("Demo nhận diện hình ảnh")
st.write("Kéo thả ảnh vào đây:")

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Show uploaded image
#     img = Image.open(uploaded_file)
#     st.image(img, caption='Tải lên', use_container_width=True)
#
#     # Preprocess and predict
#     img_array = preprocess_image(uploaded_file)
#     prediction = model.predict(img_array)
#     predicted_class = class_names[np.argmax(prediction)]
#
#     st.markdown(f"### 🎯 Kết quả: **{predicted_class}**")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    img_array = preprocess_image(uploaded_file)

    raw_preds = model.predict(img_array)[0]  # Predict raw
    predictions = tf.nn.softmax(raw_preds).numpy()  # Apply softmax

    top_3_indices = predictions.argsort()[-3:][::-1]
    top_3_labels = [(class_names[i], predictions[i]) for i in top_3_indices]

    st.subheader("🎯 Top 3 Predictions:")
    for label, score in top_3_labels:
        st.write(f"- **{label}**: {score * 100:.2f}%")