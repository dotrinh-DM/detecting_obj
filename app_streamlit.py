import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from utils import preprocess_image
from PIL import Image
import altair as alt
import io

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

# Biến để tổng hợp summary
good_predictions = 0
bad_predictions = 0

# Danh sách lưu toàn bộ kết quả để gom CSV tổng
all_predictions = []

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

            # Tạo bảng kết quả cho ảnh hiện tại
            df_result = pd.DataFrame({
                "Image Name": [uploaded_file.name]*top_n,
                "Label": [label for label, _ in top_n_labels],
                "Confidence (%)": [score*100 for _, score in top_n_labels]
            })

            # Lưu kết quả vào danh sách tổng
            all_predictions.append(df_result)

            # Kiểm tra confidence cao nhất
            if top_n_labels[0][1] * 100 < 60:
                st.warning(f"⚠️ Low confidence for {uploaded_file.name}: Hard to predict! Highest = {top_n_labels[0][1]*100:.2f}%")
                bad_predictions += 1
            else:
                good_predictions += 1

            st.subheader(f"🎯 Top {top_n} Predictions for {uploaded_file.name}:")
            st.table(df_result)

            # Vẽ bar chart
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

    # Kết thúc Predict tất cả ảnh
    st.success("✅ All Predictions Completed!")

    # 📦 Gom tất cả thành 1 bảng lớn
    df_all_predictions = pd.concat(all_predictions, ignore_index=True)

    # 🎯 Filter Confidence
    st.subheader("🔍 Filter Predictions by Confidence Threshold")
    confidence_threshold = st.slider('Select minimum confidence (%) to display', 0, 100, 0)

    df_filtered = df_all_predictions[df_all_predictions['Confidence (%)'] >= confidence_threshold]

    # Preview kết quả sau filter
    st.write(f"📊 Showing predictions with confidence >= {confidence_threshold}%:")
    st.dataframe(df_filtered)

    # Download CSV sau khi lọc
    csv_total_filtered = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Predictions (CSV)",
        data=csv_total_filtered,
        file_name="filtered_predictions_summary.csv",
        mime='text/csv',
    )

    # Tổng kết summary dashboard
    st.subheader("📊 Summary Dashboard:")

    df_summary = pd.DataFrame({
        'Prediction Type': ['Good Prediction (>=60%)', 'Low Confidence (<60%)'],
        'Number of Images': [good_predictions, bad_predictions]
    })

    st.table(df_summary)

    chart_summary = alt.Chart(df_summary).mark_bar().encode(
        x='Prediction Type',
        y='Number of Images',
        color='Prediction Type',
        tooltip=['Prediction Type', 'Number of Images']
    ).properties(width=500, height=300)

    st.altair_chart(chart_summary, use_container_width=True)