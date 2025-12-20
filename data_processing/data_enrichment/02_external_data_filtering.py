import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

def calculate_change_ratio(mask_path):
    """
    Tính toán tỷ lệ diện tích thay đổi (pixel trắng) trên tổng diện tích của ảnh mask.
    """
    mask = cv2.imread(mask_path, 0)
    return np.count_nonzero(mask) / mask.size if mask is not None else 0

def process_filtering(config):
    """
    Thực hiện lọc các cặp ảnh thỏa mãn ngưỡng thay đổi và sao chép sang thư mục đích với tên mới.
    """
    os.makedirs(config['dst_a'], exist_ok=True)
    os.makedirs(config['dst_b'], exist_ok=True)
    os.makedirs(config['dst_l'], exist_ok=True)
    
    files = [f for f in os.listdir(config['src_l']) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    count = 0
    
    for fname in tqdm(files, desc=f"Filtering {config['name']}"):
        ratio = calculate_change_ratio(os.path.join(config['src_l'], fname))
        if ratio >= config['threshold']:
            new_name = f"cdd_{fname}"
            shutil.copy(os.path.join(config['src_a'], fname), os.path.join(config['dst_a'], new_name))
            shutil.copy(os.path.join(config['src_b'], fname), os.path.join(config['dst_b'], new_name))
            shutil.copy(os.path.join(config['src_l'], fname), os.path.join(config['dst_l'], new_name))
            count += 1
    print(f"{config['name']}: Đã lấy {count} ảnh (Ngưỡng > {config['threshold']*100}%)")

if __name__ == "__main__":
    # Cấu hình lọc cho tập TRAIN (Ngưỡng cao để cân bằng)
    train_config = {
        'name': 'CDD_Train',
        'threshold': 0.25,
        'src_a': '/kaggle/input/cddchangedetection/subset/train/A',
        'src_b': '/kaggle/input/cddchangedetection/subset/train/B',
        'src_l': '/kaggle/input/cddchangedetection/subset/train/OUT',
        'dst_a': '/kaggle/working/supplement_data/A',
        'dst_b': '/kaggle/working/supplement_data/B',
        'dst_l': '/kaggle/working/supplement_data/OUT'
    }
    
    # Cấu hình lọc cho tập TEST (Ngưỡng thấp hơn để đánh giá đa dạng)
    test_config = {
        'name': 'CDD_Test',
        'threshold': 0.05,
        'src_a': '/kaggle/input/cddchangedetection/subset/test/A',
        'src_b': '/kaggle/input/cddchangedetection/subset/test/B',
        'src_l': '/kaggle/input/cddchangedetection/subset/test/OUT',
        'dst_a': '/kaggle/working/cdd_test_filtered/A',
        'dst_b': '/kaggle/working/cdd_test_filtered/B',
        'dst_l': '/kaggle/working/cdd_test_filtered/label'
    }
    
    process_filtering(train_config)
    process_filtering(test_config)