import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class CatDogEfficientNetB0(nn.Module):
    def __init__(self):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.base = efficientnet_b0(weights=weights)
        for param in self.base.parameters():
            param.requires_grad = False
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.base(x)
