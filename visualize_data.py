import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def count_images(data_dir):
    """Đếm số lượng ảnh trong mỗi lớp và tập dữ liệu"""
    counts = {
        'train': {'cat': 0, 'dog': 0},
        'test': {'cat': 0, 'dog': 0}
    }
    
    for split in ['train', 'test']:
        for class_name in ['cat', 'dog']:
            path = os.path.join(data_dir, split, class_name)
            counts[split][class_name] = len(os.listdir(path))
    
    return counts

def plot_sample_images(data_dir, num_samples=5):
    """Hiển thị một số ảnh mẫu ngẫu nhiên từ mỗi lớp"""
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for idx, class_name in enumerate(['cat', 'dog']):
        # Lấy danh sách tất cả ảnh từ cả train và test
        train_path = os.path.join(data_dir, 'train', class_name)
        test_path = os.path.join(data_dir, 'test', class_name)
        all_images = os.listdir(train_path) + os.listdir(test_path)
        
        # Chọn ngẫu nhiên các ảnh
        sample_images = random.sample(all_images, num_samples)
        
        for i, img_name in enumerate(sample_images):
            # Tìm ảnh trong train hoặc test
            if img_name in os.listdir(train_path):
                img_path = os.path.join(train_path, img_name)
            else:
                img_path = os.path.join(test_path, img_name)
                
            img = Image.open(img_path)
            axes[idx, i].imshow(img)
            axes[idx, i].axis('off')
            axes[idx, i].set_title(f'{class_name} {i+1}')
    
    plt.tight_layout()
    plt.show()

def analyze_image_sizes(data_dir):
    """Phân tích kích thước của ảnh"""
    widths = []
    heights = []
    
    for split in ['train', 'test']:
        for class_name in ['cat', 'dog']:
            path = os.path.join(data_dir, split, class_name)
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                except:
                    continue
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.hist(widths, bins=50)
    plt.title('Phân phối chiều rộng ảnh')
    plt.xlabel('Chiều rộng (pixels)')
    plt.ylabel('Số lượng ảnh')
    
    plt.subplot(122)
    plt.hist(heights, bins=50)
    plt.title('Phân phối chiều cao ảnh')
    plt.xlabel('Chiều cao (pixels)')
    plt.ylabel('Số lượng ảnh')
    
    plt.tight_layout()
    plt.show()

def analyze_color_distribution(data_dir, num_samples=100):
    """Phân tích phân phối màu sắc"""
    r_vals = []
    g_vals = []
    b_vals = []
    
    for split in ['train', 'test']:
        for class_name in ['cat', 'dog']:
            path = os.path.join(data_dir, split, class_name)
            all_images = os.listdir(path)
            sample_images = random.sample(all_images, min(num_samples, len(all_images)))
            
            for img_name in sample_images:
                img_path = os.path.join(path, img_name)
                try:
                    img = np.array(Image.open(img_path))
                    r_vals.extend(img[:,:,0].flatten())
                    g_vals.extend(img[:,:,1].flatten())
                    b_vals.extend(img[:,:,2].flatten())
                except:
                    continue
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.hist(r_vals, bins=50, color='red', alpha=0.5)
    plt.title('Phân phối kênh Red')
    plt.xlabel('Giá trị pixel')
    
    plt.subplot(132)
    plt.hist(g_vals, bins=50, color='green', alpha=0.5)
    plt.title('Phân phối kênh Green')
    plt.xlabel('Giá trị pixel')
    
    plt.subplot(133)
    plt.hist(b_vals, bins=50, color='blue', alpha=0.5)
    plt.title('Phân phối kênh Blue')
    plt.xlabel('Giá trị pixel')
    
    plt.tight_layout()
    plt.show()

def main():
    data_dir = './data'
    
    # 1. Đếm số lượng ảnh
    counts = count_images(data_dir)
    print("\nSố lượng ảnh trong dataset:")
    print(f"Train set: {counts['train']['cat']} ảnh mèo, {counts['train']['dog']} ảnh chó")
    print(f"Test set: {counts['test']['cat']} ảnh mèo, {counts['test']['dog']} ảnh chó")
    
    # 2. Hiển thị ảnh mẫu
    print("\nHiển thị một số ảnh mẫu...")
    plot_sample_images(data_dir)
    
    # 3. Phân tích kích thước ảnh
    print("\nPhân tích kích thước ảnh...")
    analyze_image_sizes(data_dir)
    
    # 4. Phân tích phân phối màu sắc
    print("\nPhân tích phân phối màu sắc...")
    analyze_color_distribution(data_dir)

if __name__ == '__main__':
    main()