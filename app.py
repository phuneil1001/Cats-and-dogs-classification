import streamlit as st
from predict_resnet18 import predict_image
import tempfile
from PIL import Image

st.set_page_config(page_title="Phân loại Chó/Mèo", layout="centered")
st.title("🐾 Ứng dụng phân loại ảnh Chó và Mèo")
st.write("Tải ảnh lên để xem dự đoán mô hình phân loại là **Chó** hay **Mèo** 🐶🐱")

# Upload ảnh
uploaded_file = st.file_uploader("📷 Tải ảnh lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Hiển thị ảnh
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

        # Lưu tạm ảnh để truyền đường dẫn vào hàm predict_image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name

        # Dự đoán
        with st.spinner("🔍 Đang phân tích..."):
            result, confidence = predict_image(tmp_path)
            st.success(f"✅ Kết quả: **{result}**")
            st.info(f"🔒 Độ tin cậy: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
