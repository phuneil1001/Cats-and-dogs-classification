import torch
from torchvision import transforms
from PIL import Image
from model_efficientnet import CatDogEfficientNetB0

# Đường dẫn model và class
MODEL_PATH = 'efficientnet_model.pth'
CLASS_NAMES = ['cat', 'dog']  # Sửa lại nếu class khác

# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # Thêm batch dimension
    return img

# Hàm dự đoán
def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogEfficientNetB0()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    img = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        predicted_class = CLASS_NAMES[pred.item()]
        confidence = conf.item() * 100
    return predicted_class, confidence

if __name__ == "__main__":
    image_path = input("Nhập đường dẫn ảnh cần dự đoán: ")
    result, confidence = predict(image_path)
    print(f"Ảnh này là: {result} ({confidence:.2f}%)")