"""
Tập tin huấn luyện mô hình phân loại nhanh ảnh Chó và Mèo
---------------------------------------------------------
Mô tả:
File này triển khai quy trình huấn luyện phiên bản tối ưu và nhẹ hơn của
mô hình phân loại hình ảnh chó và mèo. Thiết kế để chạy nhanh và hiệu quả
với tập dữ liệu nhỏ hơn, số epoch ít hơn và mô hình đơn giản hơn.

Đặc điểm chính:
1. Sử dụng mô hình nhẹ (model_fast.py) với ít tham số hơn so với mô hình chính
2. Dùng tập dữ liệu rút gọn (500 ảnh train, 200 ảnh validation) để huấn luyện nhanh
3. Binary classification trực tiếp (1 output) thay vì 2 lớp như mô hình chính
4. Ít epoch hơn (3 thay vì 10) nhằm giảm thời gian huấn luyện
5. Sử dụng BCEWithLogitsLoss và sigmoid phù hợp cho bài toán phân loại nhị phân

Quy trình huấn luyện:
- Mô hình được huấn luyện qua 3 epochs với dữ liệu đã rút gọn
- Đánh giá độ chính xác trên tập validation sau mỗi epoch
- Lưu mô hình có độ chính xác validation cao nhất
- Đánh giá cuối cùng trên tập test khi huấn luyện hoàn tất
"""

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
