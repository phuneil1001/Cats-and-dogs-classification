import torch
import torch.nn as nn

class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        
        # Các lớp tích chập (Convolutional layers)
        # Gồm 3 khối, mỗi khối có:
        # - Lớp tích chập (Conv2d)
        # - Hàm kích hoạt ReLU
        # - Lớp pooling để giảm kích thước
        self.conv_layers = nn.Sequential(
            # Khối 1: đầu vào 3 kênh (RGB) -> 32 feature maps
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Khối 2: 32 -> 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Khối 3: 64 -> 128 feature maps
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Các lớp fully connected
        self.fc_layers = nn.Sequential(
            # Làm phẳng dữ liệu
            nn.Flatten(),
            # FC 1: 128*28*28 -> 512 neurons
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            # Dropout để tránh overfitting
            nn.Dropout(0.5),
            # FC 2: 512 -> 2 neurons (cat/dog)
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        # Truyền dữ liệu qua các lớp tích chập
        x = self.conv_layers(x)
        # Truyền qua các lớp fully connected
        x = self.fc_layers(x)
        return x
