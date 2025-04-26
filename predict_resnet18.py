import torch
from torchvision import transforms
from PIL import Image
from model_resnet18 import CatDogClassifier  # dùng ResNet18
import json
import os

# Định nghĩa transform giống như lúc huấn luyện với ResNet18
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Hàm dự đoán ảnh
def predict_image(image_path):
    try:
        # Kiểm tra tệp tồn tại
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

        # Tải model ResNet18
        model = CatDogClassifier()
        model.load_state_dict(torch.load("cat_dog_resnet18_ver2.pth", map_location=torch.device("cpu")))
        model.eval()

        # Load class_to_idx
        with open("class_to_idx.json", "r") as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Xử lý ảnh
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

        # Dự đoán
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, pred = torch.max(probs, 1)
            confidence = probs[0][pred.item()].item()

        label = idx_to_class[pred.item()]
        emoji = "🐱" if "cat" in label.lower() else "🐶"
        return f"{label.capitalize()} {emoji}", confidence * 100

    except Exception as e:
        raise RuntimeError(f"Lỗi khi dự đoán: {str(e)}")

# Chạy thử khi chạy trực tiếp
if __name__ == "__main__":
    image_path = r"C:\Users\ADMIN\Desktop\Xulyanh2\data\test\cat\1359.jpg"
    try:
        result, confidence = predict_image(image_path)
        print(f"Kết quả: {result} (độ tin cậy: {confidence:.2f}%)")
    except Exception as e:
        print(f"Lỗi: {e}")
