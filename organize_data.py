import os
import shutil
import random

def organize_dataset(source_dir='PetImages', dest_dir='data', train_ratio=0.7, val_ratio=0.15):
    # Tạo các thư mục cần thiết
    for folder in ['train/cat', 'train/dog', 'test/cat', 'test/dog', 'val/cat', 'val/dog']:
        os.makedirs(os.path.join(dest_dir, folder), exist_ok=True)

    # Xử lý ảnh mèo
    cat_files = os.listdir(os.path.join(source_dir, 'Cat'))
    random.shuffle(cat_files)  # Xáo trộn ngẫu nhiên danh sách file
    
    # Tính toán các điểm chia
    train_split = int(len(cat_files) * train_ratio)
    val_split = int(len(cat_files) * (train_ratio + val_ratio))

    # Di chuyển ảnh mèo vào train, val và test
    for i, file in enumerate(cat_files):
        src = os.path.join(source_dir, 'Cat', file)
        if os.path.isfile(src):  # Kiểm tra file có tồn tại không
            try:
                if i < train_split:
                    dst = os.path.join(dest_dir, 'train', 'cat', file)
                elif i < val_split:
                    dst = os.path.join(dest_dir, 'val', 'cat', file)
                else:
                    dst = os.path.join(dest_dir, 'test', 'cat', file)
                shutil.copy2(src, dst)
            except Exception as e:
                print(f'Lỗi khi copy file {file}: {str(e)}')

    # Xử lý ảnh chó
    dog_files = os.listdir(os.path.join(source_dir, 'Dog'))
    random.shuffle(dog_files)  # Xáo trộn ngẫu nhiên danh sách file
    
    # Tính toán các điểm chia
    train_split = int(len(dog_files) * train_ratio)
    val_split = int(len(dog_files) * (train_ratio + val_ratio))

    # Di chuyển ảnh chó vào train, val và test
    for i, file in enumerate(dog_files):
        src = os.path.join(source_dir, 'Dog', file)
        if os.path.isfile(src):  # Kiểm tra file có tồn tại không
            try:
                if i < train_split:
                    dst = os.path.join(dest_dir, 'train', 'dog', file)
                elif i < val_split:
                    dst = os.path.join(dest_dir, 'val', 'dog', file)
                else:
                    dst = os.path.join(dest_dir, 'test', 'dog', file)
                shutil.copy2(src, dst)
            except Exception as e:
                print(f'Lỗi khi copy file {file}: {str(e)}')

if __name__ == '__main__':
    print("Bắt đầu tổ chức lại dữ liệu...")
    # Sử dụng tỉ lệ 70:15:15 cho train:val:test
    organize_dataset(train_ratio=0.7, val_ratio=0.15)
    print("Hoàn thành tổ chức lại dữ liệu!")