import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        
        # Sử dụng pretrained weights chuẩn (ImageNet)
        weights = ResNet18_Weights.DEFAULT
        self.base_model = resnet18(weights=weights)

        # Đóng băng toàn bộ layer (chỉ fine-tune fc layer)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Thay thế lớp fully connected cuối bằng lớp phân loại 2 lớp
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.base_model(x)
