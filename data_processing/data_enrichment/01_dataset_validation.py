import os
import cv2
import numpy as np
from termcolor import colored
from tqdm import tqdm

VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')

def get_clean_file_list(folder_path):
    """
    Lấy danh sách các file ảnh hợp lệ từ một thư mục, loại bỏ file ẩn và file rác.
    """
    if not os.path.exists(folder_path):
        return set()
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTS)]
    return set(files)

def check_consistency(paths):
    """
    Kiểm tra sự đồng bộ về số lượng và tên file giữa các thư mục A, B và Label.
    """
    files_A = get_clean_file_list(paths["A"])
    files_B = get_clean_file_list(paths["B"])
    files_L = get_clean_file_list(paths["Label"])
    
    mismatch_AB = files_A.symmetric_difference(files_B)
    mismatch_AL = files_A.symmetric_difference(files_L)
    return files_A, mismatch_AB, mismatch_AL

def analyze_change_distribution(label_dir):
    """
    Thống kê tỷ lệ diện tích thay đổi (pixel trắng) trong các ảnh nhãn để đánh giá độ cân bằng.
    """
    image_files = [f for f in os.listdir(label_dir) if f.lower().endswith(VALID_EXTS)]
    stats = {"high": 0, "medium": 0, "low": 0, "total": 0}
    
    for filename in tqdm(image_files):
        img = cv2.imread(os.path.join(label_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        ratio = (np.count_nonzero(img) / img.size) * 100
        if ratio >= 75: stats["high"] += 1
        elif ratio <= 25: stats["low"] += 1
        else: stats["medium"] += 1
        stats["total"] += 1
    return stats