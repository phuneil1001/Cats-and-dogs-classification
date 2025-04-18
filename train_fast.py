import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model_fast import get_model  # mô hình nhẹ hơn
from dataset_prep_fast import get_data_loaders  # phiên bản dataset rút gọn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DATA_DIR = './data'

def train():
    train_loader, val_loader, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE)

    model = get_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = (torch.sigmoid(outputs) > 0.5).int()
            correct += (predicted == labels.int()).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        train_acc = correct / total
        val_acc = evaluate(model, val_loader)
        print(f"✅ Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_fast.pth")
            print("💾 Đã lưu mô hình tốt nhất!")

    # Test
    print("\n🧪 Đánh giá trên test set:")
    model.load_state_dict(torch.load("best_model_fast.pth"))
    test_acc = evaluate(model, test_loader)
    print(f"🎯 Test Accuracy: {test_acc:.4f}")

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).int()
            correct += (predicted == labels.int()).sum().item()
            total += labels.size(0)
    return correct / total

if __name__ == "__main__":
    train()
