import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os

# Hàm kiểm tra ảnh lỗi
def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
            img = Image.open(filepath).convert('RGB')  # thử load RGB luôn
        return True
    except:
        print(f"[!] Ảnh lỗi hoặc không hợp lệ: {filepath}")
        return False

# Hàm dọn dữ liệu lỗi trong thư mục
def clean_dataset(directory):
    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if not is_valid_image(img_path):
                    os.remove(img_path)

# Gọi dọn ảnh lỗi trước khi tạo dataset
def get_data_loaders(data_dir='./data', batch_size=32):
    print("🧹 Đang kiểm tra và loại bỏ ảnh lỗi...")
    clean_dataset(os.path.join(data_dir, 'train'))
    clean_dataset(os.path.join(data_dir, 'val'))
    clean_dataset(os.path.join(data_dir, 'test'))

    # Transform đúng chuẩn cho ResNet
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("📂 Nhãn lớp:", train_dataset.classes)
    print(f"🖼️ Số lượng ảnh: train = {len(train_dataset)}, val = {len(val_dataset)}, test = {len(test_dataset)}")

    return train_loader, val_loader, test_loader
