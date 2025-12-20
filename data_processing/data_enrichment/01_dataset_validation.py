import os
import cv2
import random
import numpy as np
from termcolor import colored
from tqdm import tqdm

def get_clean_file_list(folder_path):
    """
    Lấy danh sách các tệp ảnh hợp lệ, loại bỏ các tệp ẩn hoặc tệp không đúng định dạng ảnh.
    """
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    if not os.path.exists(folder_path):
        return set()
    return set([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])

def check_consistency(paths):
    """
    Kiểm tra sự đồng bộ 1-1 về tên file và kích thước ảnh giữa các thư mục Before, After và Label.
    """
    files_a = get_clean_file_list(paths["A"])
    files_b = get_clean_file_list(paths["B"])
    files_l = get_clean_file_list(paths["Label"])
    
    is_consistent = len(files_a) == len(files_b) == len(files_l) and not files_a.symmetric_difference(files_b)
    
    if is_consistent and len(files_a) > 0:
        sample_f = random.choice(list(files_a))
        img_a = cv2.imread(os.path.join(paths["A"], sample_f))
        img_l = cv2.imread(os.path.join(paths["Label"], sample_f), 0)
        if img_a.shape[:2] != img_l.shape:
            return False, "Size Mismatch"
    return is_consistent, len(files_a)

def analyze_distribution(label_dir):
    """
    Thống kê tỷ lệ pixel thay đổi trên toàn bộ tập dữ liệu để xác định mức độ mất cân bằng class.
    """
    files = get_clean_file_list(label_dir)
    stats = {"high": 0, "medium": 0, "low": 0}
    
    for fname in tqdm(files, desc="Analyzing Distribution"):
        img = cv2.imread(os.path.join(label_dir, fname), cv2.IMREAD_GRAYSCALE)
        ratio = (np.count_nonzero(img) / img.size) * 100
        if ratio >= 75: stats["high"] += 1
        elif ratio <= 25: stats["low"] += 1
        else: stats["medium"] += 1
    return stats

if __name__ == "__main__":
    LEVIR_PATHS = {
        "A": "/kaggle/input/levir-processed/kaggle/working/processed_balanced_levircd/train/A",
        "B": "/kaggle/input/levir-processed/kaggle/working/processed_balanced_levircd/train/B",
        "Label": "/kaggle/input/levir-processed/kaggle/working/processed_balanced_levircd/train/label"
    }
    
    print("KIỂM TRA ĐỒNG BỘ")
    consistent, count = check_consistency(LEVIR_PATHS)
    print(colored(f"Kết quả: {'OK' if consistent else 'LỖI'} | Tổng: {count} ảnh", 'green' if consistent else 'red'))
    
    print("\nTHỐNG KÊ PHÂN PHỐI")
    dist = analyze_distribution(LEVIR_PATHS["Label"])
    for k, v in dist.items():
        print(f"Mức {k.upper()}: {v} ảnh ({v/count*100:.2f}%)")