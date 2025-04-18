import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
            # Try to load it as RGB to catch other potential issues
            img = Image.open(filepath).convert('RGB')
        return True
    except:
        print(f"Corrupted or invalid image found: {filepath}")
        return False

class ValidImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        self.root = root
        # First, filter out invalid images
        valid_files = []
        for class_dir in os.listdir(root):
            class_path = os.path.join(root, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if is_valid_image(img_path):
                        valid_files.append(img_path)
        
        super(ValidImageFolder, self).__init__(root=root, transform=transform)

def get_data_loaders(data_dir='./data', batch_size=32):
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
    
    # Use ValidImageFolder instead of datasets.ImageFolder
    train_dataset = ValidImageFolder(root=data_dir + '/train', transform=train_transform)
    val_dataset = ValidImageFolder(root=data_dir + '/val', transform=val_transform)
    test_dataset = ValidImageFolder(root=data_dir + '/test', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Classes:", train_dataset.classes)
    print(f"Tổng ảnh train: {len(train_dataset)} | val: {len(val_dataset)} | test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
