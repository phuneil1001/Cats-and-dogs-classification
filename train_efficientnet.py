import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_efficientnet import CatDogEfficientNetB0  
from tqdm import tqdm  # Thêm tqdm để hiển thị tiến trình

# Cấu hình
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

# Tiền xử lý dữ liệu
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('data/train', transform=transform)
val_dataset = datasets.ImageFolder('data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load mô hình EfficientNet từ file model_efficientnet.py
model = CatDogEfficientNetB0()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
# optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
best_acc = 0.0  # Biến lưu val acc tốt nhất

# Train loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        train_bar.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

    # Đánh giá trên tập validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}")

    # Lưu checkpoint nếu val acc tốt nhất
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'efficientnet_best.pth')
        print(f"==> Đã lưu model tốt nhất với val acc: {best_acc:.4f}")

# Lưu model cuối cùng
torch.save(model.state_dict(), 'efficientnet_model_final.pth')