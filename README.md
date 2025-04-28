📦 Quy trình sử dụng cụ thể (bạn chỉ cần làm theo)
==================================================

| Bước | Việc làm | Câu lệnh mẫu |
| --- | --- | --- |
| 1 | Clone về | `mkdir detecting_obj` |
| 2 | Cài thư viện | `pip install tensorflow matplotlib numpy` |
| 3 | Chạy file huấn luyện | `python train.py` |
| 4 | Copy 1 ảnh mới vào cùng thư mục (ví dụ: `img/dog.jpg`) |  |
| 5 | Chỉnh sửa dòng đường dẫn trong `predict.py` thành `'img/dog.jpg'` |  |
| 6 | Chạy dự đoán | `python predict.py` |
| 7 | Xem kết quả trên console: "Predicted: dog" |  |

# **Tính năng nâng cao:**
* Upload nhiều ảnh
* Predict từng ảnh
* Top-N Slider linh động
* Cảnh báo confidence thấp
* Download CSV
* Summary Dashboard + Chart