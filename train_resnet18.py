import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model_resnet18 import CatDogClassifier
from dataset_prep_resnet18 import get_data_loaders
import copy
import os
import json

# --- Cấu hình ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DATA_DIR = './data'
PATIENCE = 7  # cho EarlyStopping

def train():
    # Tải dữ liệu
    train_loader, val_loader, test_loader = get_data_loaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE)

    # Khởi tạo mô hình (ResNet18)
    model = CatDogClassifier().to(DEVICE)

    # Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Lưu class_to_idx để dùng khi predict
    class_to_idx = train_loader.dataset.class_to_idx  # lấy từ ImageFolder
    with open("class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\n📘 Epoch [{epoch+1}/{EPOCHS}]")
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100. * correct / total
        avg_train_loss = running_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"✅ Train Acc: {train_acc:.2f}% | Loss: {avg_train_loss:.4f}")
        print(f"🧪 Val Acc:   {val_acc:.2f}% | Loss: {avg_val_loss:.4f}")

        # --- ModelCheckpoint ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'cat_dog_resnet18_ver2.pth')
            print("💾 Đã lưu mô hình tốt nhất!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"📌 Không cải thiện ({epochs_no_improve}/{PATIENCE})")

        # --- EarlyStopping ---
        if epochs_no_improve >= PATIENCE:
            print("⏹️ Dừng sớm do không cải thiện validation accuracy.")
            break

    print(f"\n🎯 Huấn luyện hoàn tất. Val Acc tốt nhất: {best_val_acc:.2f}%")

    # --- Test ---
    model.load_state_dict(best_model_wts)
    test_acc = evaluate(model, test_loader)
    print(f"📊 Test Accuracy: {test_acc:.2f}%")

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100. * correct / total

if __name__ == '__main__':
    train()
