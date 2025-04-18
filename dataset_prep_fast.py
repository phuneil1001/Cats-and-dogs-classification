from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

def get_data_loaders(data_dir='./data', batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=data_dir + '/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=data_dir + '/val', transform=transform)
    test_dataset = datasets.ImageFolder(root=data_dir + '/test', transform=transform)

    # Dùng 500 ảnh đầu để huấn luyện nhanh
    train_dataset = Subset(train_dataset, range(500))
    val_dataset = Subset(val_dataset, range(200))
    test_dataset = Subset(test_dataset, range(200))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
