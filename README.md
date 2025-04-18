---
title: "Phân loại ảnh Chó Mèo 🐶🐱"
emoji: "🐾"
colorFrom: "gray"
colorTo: "purple"
sdk: streamlit
sdk_version: "1.44.1"
app_file: app.py
pinned: true
---

# 🐾 Ứng dụng Phân loại ảnh Chó và Mèo bằng ResNet18

🎯 **Mục tiêu**: Dự đoán ảnh bạn tải lên là **Chó 🐶** hay **Mèo 🐱** dựa trên mô hình học sâu **ResNet18**.

🚀 Ứng dụng được xây dựng bằng **Streamlit + PyTorch**, triển khai trên nền tảng **Hugging Face Spaces**.

---

## 📥 Hướng dẫn sử dụng

1. Nhấn **"Browse files"** hoặc kéo thả ảnh vào ô tương tác
2. Ứng dụng sẽ xử lý ảnh và hiển thị kết quả phân loại
3. Bạn sẽ thấy tên loài vật (Chó hoặc Mèo) và độ tin cậy của dự đoán

📌 **Định dạng ảnh hỗ trợ**: JPG, JPEG, PNG  
📏 **Kích thước ảnh đầu vào**: Tự động resize về 224x224 (chuẩn đầu vào của ResNet)

---

## 🧠 Về mô hình ResNet18

- Dựa trên kiến trúc **Residual Network (ResNet18)** nổi tiếng của Microsoft
- Được fine-tune từ mô hình pretrained trên ImageNet
- Gồm các khối residual giúp mô hình học tốt hơn trên ảnh có nhiều đặc điểm phức tạp
- Output là softmax 2 lớp: Chó 🐶 hoặc Mèo 🐱

---

## 🛠️ Công nghệ sử dụng

| Thành phần     | Mô tả                          |
|----------------|-------------------------------|
| `Streamlit`    | Tạo giao diện web đơn giản     |
| `PyTorch`      | Load và chạy mô hình ResNet18 |
| `TorchVision`  | Tiền xử lý ảnh (transform, normalize) |
| `Pillow`       | Đọc ảnh định dạng PNG/JPG      |

---

## 👨‍💻 Tác giả

> ✨ Ứng dụng được phát triển bởi [Phuneil](https://huggingface.co/Phuneil) – kỹ sư phần mềm yêu thích lập trình nhúng và AI ứng dụng thực tế.

---

## 🌟 Hãy để lại 1 ⭐ nếu bạn thấy dự án này hữu ích!
