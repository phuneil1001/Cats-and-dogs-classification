import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import CatDogClassifier
from dataset_prep import get_data_loaders

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DATA_DIR = './data'

def train():
    # Tải dữ liệu
    train_loader, val_loader, test_loader = get_data_loaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE)

    # Khởi tạo mô hình
    model = CatDogClassifier()
    model.to(DEVICE)

    # Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            loop.set_postfix(loss=running_loss/len(loop), acc=100.*correct/total)

        train_acc = 100. * correct / total

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f'Epoch [{epoch+1}/{EPOCHS}]:')
        print(f'Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

        # Lưu model tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'cat_dog_classifier.pth')
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')

    print("Training completed!")

if __name__ == '__main__':
    train()
