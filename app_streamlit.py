import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from click import style

from utils import preprocess_image
from PIL import Image
import altair as alt
import io

# Load model
model = tf.keras.models.load_model('cnn_cifar10_model.h5')

# Class names CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.title("Nhận diện hình ảnh bằng AI")
st.write("Vui lòng kéo thả hoặc chọn nhiều ảnh để nhận diện.")

st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50; /* Nút thường */
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
    }
    .stDownloadButton > button {
        background-color: #2196F3; /* Nút download */
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
    }
     .stDownloadButton > button:hover {
        background-color: #0b7dff; /* Màu nền khi hover (đậm hơn) */
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Top-N Slider
top_n = st.slider('Chọn các label dự đoán', min_value=1, max_value=10, value=3)

uploaded_files = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Biến để tổng hợp summary
good_predictions = 0
bad_predictions = 0

# Danh sách lưu toàn bộ kết quả để gom CSV tổng
all_predictions = []

if uploaded_files:
    with st.spinner('Đang đoán...'):
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
                "Độ tự tin (%)": [score * 100 for _, score in top_n_labels]
            })

            # Lưu kết quả vào danh sách tổng
            all_predictions.append(df_result)

            # Kiểm tra confidence cao nhất
            if top_n_labels[0][1] * 100 < 60:
                st.warning(f"⚠️ Độ tự tin thấp cho ảnh {uploaded_file.name}: Khó dự đoán! Cao nhất = {top_n_labels[0][1] * 100:.2f}%")
                bad_predictions += 1
            else:
                good_predictions += 1

            st.subheader(f"🎯 Top {top_n} kết quả dự đoán cho ảnh {uploaded_file.name}:")
            st.table(df_result)

            # Vẽ bar chart
            df_chart = pd.DataFrame({
                'Label': [label for label, _ in top_n_labels],
                'Độ tự tin': [score * 100 for _, score in top_n_labels]
            })

            chart = alt.Chart(df_chart).mark_bar().encode(
                x=alt.X('Label', sort='-y'),
                y='Độ tự tin',
                color=alt.Color('Label', legend=None),
                tooltip=['Label', 'Độ tự tin']
            ).properties(width=400, height=300)

            st.altair_chart(chart, use_container_width=True)

    # Kết thúc Predict tất cả ảnh
    st.success("✅ Hoàn thành!")
    st.markdown("------------------------------")
    st.header("THỐNG KÊ SỐ LIỆU")

    # 📦 Gom tất cả thành 1 bảng lớn
    df_all_predictions = pd.concat(all_predictions, ignore_index=True)

    # 🎯 Filter Độ tự tin
    st.subheader("Lọc kết quả dự đoán theo mức độ tự tin toàn bộ ảnh")
    confidence_threshold = st.slider('Select minimum confidence (%) to display', 0, 100, 0)

    df_filtered = df_all_predictions[df_all_predictions['Độ tự tin (%)'] >= confidence_threshold]

    # Preview kết quả sau filter
    st.write(f"Hiển thị các dự đoán có độ tự tin >= {confidence_threshold}%:")
    st.dataframe(df_filtered)

    # Download CSV sau khi lọc
    csv_total_filtered = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Tải xuống (CSV)",
        data=csv_total_filtered,
        file_name="filtered_predictions_summary.csv",
        mime='text/csv',
    )

    # Tổng kết summary dashboard
    st.subheader("Bảng tổng kết kết quả dự đoán toàn bộ ảnh")

    df_summary = pd.DataFrame({
        'Loại dự đoán': ['Dự đoán tốt (>=60%)', 'Độ tự tin thấp (<60%)'],
        'Số lượng ảnh': [good_predictions, bad_predictions]
    })

    st.table(df_summary)

    chart_summary = alt.Chart(df_summary).mark_bar().encode(
        x='Loại dự đoán',
        y='Số lượng ảnh',
        color='Loại dự đoán',
        tooltip=['Loại dự đoán', 'Số lượng ảnh']
    ).properties(width=500, height=600)

    st.altair_chart(chart_summary, use_container_width=True)

    # Biểu đồ phân phối toàn bộ độ tự tin
    st.subheader("Biểu đồ phân phối các mức độ tự tin toàn bộ ảnh")
    st.info("Gợi ý: Nếu mô hình tốt, biểu đồ sẽ lệch phải (nhiều dự đoán trên 80%-100%)")

    # Histogram phân phối
    hist_chart = alt.Chart(df_all_predictions).mark_bar().encode(
        alt.X("Độ tự tin (%)", bin=alt.Bin(maxbins=30), title='Độ tự tin (%)'),
        y='count()',
        tooltip=['count()']
    ).properties(
        width=600,
        height=400
    )

    st.altair_chart(hist_chart, use_container_width=True)