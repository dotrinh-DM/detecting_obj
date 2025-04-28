import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from utils import preprocess_image
from PIL import Image
import altair as alt

# Load model
model = tf.keras.models.load_model('cnn_cifar10_model.h5')

# Class names CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("🚀 Image Recognition Demo")
st.write("Upload one or more images and let the model predict them!")

# Top-N Slider
top_n = st.slider('Select Top-N Predictions to Display', min_value=1, max_value=10, value=3)

uploaded_files = st.file_uploader("Choose image files...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner('🔍 Predicting... Please wait...'):
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            st.image(img, caption=f'Uploaded: {uploaded_file.name}', use_container_width=True)

            img_array = preprocess_image(uploaded_file)

            raw_preds = model.predict(img_array)[0]
            predictions = tf.nn.softmax(raw_preds).numpy()

            # lấy top-N theo slider
            top_n_indices = predictions.argsort()[-top_n:][::-1]
            top_n_labels = [(class_names[i], predictions[i]) for i in top_n_indices]

            # Tạo bảng kết quả
            df_result = pd.DataFrame({
                "Label": [label for label, _ in top_n_labels],
                "Confidence (%)": [score*100 for _, score in top_n_labels]
            })

            # Kiểm tra Confidence cao nhất
            if top_n_labels[0][1] * 100 < 60:
                st.warning(f"⚠️ Low confidence for {uploaded_file.name}: Hard to predict! Highest = {top_n_labels[0][1]*100:.2f}%")

            st.subheader(f"🎯 Top {top_n} Predictions for {uploaded_file.name}:")
            st.table(df_result)

            # Vẽ bar chart cho top-N
            df_chart = pd.DataFrame({
                'Label': [label for label, _ in top_n_labels],
                'Confidence': [score*100 for _, score in top_n_labels]
            })

            chart = alt.Chart(df_chart).mark_bar().encode(
                x=alt.X('Label', sort='-y'),
                y='Confidence',
                color=alt.Color('Label', legend=None),
                tooltip=['Label', 'Confidence']
            ).properties(width=400, height=300)

            st.altair_chart(chart, use_container_width=True)

    st.success("✅ All Predictions Completed!")