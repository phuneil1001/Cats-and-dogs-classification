import torch
from PIL import Image
from torchvision import transforms
from model import CatDogClassifier
import os

def predict_image(image_path):
    try:
        # Kiểm tra file ảnh tồn tại
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")
            
        # Kiểm tra và sử dụng GPU nếu có
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Kiểm tra file model tồn tại
        model_path = 'cat_dog_classifier.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
        
        # Tải model đã huấn luyện
        model = CatDogClassifier()
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải model: {str(e)}")
            
        model.to(device)
        model.eval()
        
        # Chuẩn bị ảnh đầu vào
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            transform = transforms.Compose([
                transforms.Resize(256),  # Resize to maintain aspect ratio
                transforms.CenterCrop(224),  # Crop center to match training size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            
            image = transform(image).unsqueeze(0).to(device)
        except Exception as e:
            raise RuntimeError(f"Lỗi khi xử lý ảnh: {str(e)}")
        
        # Thực hiện dự đoán
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            
            # Lấy xác suất dự đoán
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted.item()].item() * 100
            
            result = "mèo" if predicted.item() == 0 else "chó"
            return result, confidence
            
    except Exception as e:
        raise Exception(f"Lỗi trong quá trình dự đoán: {str(e)}")

if __name__ == '__main__':
    try:
        # Ví dụ sử dụng
        image_path = r'C:\Users\ADMIN\Desktop\Xulyanh2\data\test\cat\12084.jpg'
        result, confidence = predict_image(image_path)
        print(f'Kết quả dự đoán: {result} (độ tin cậy: {confidence:.2f}%)')
    except Exception as e:
        print(f"Lỗi: {str(e)}")
