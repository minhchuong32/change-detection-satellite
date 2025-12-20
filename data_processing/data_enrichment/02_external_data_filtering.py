"""
Xử lý tập dữ liệu phụ (CCD). Thực hiện lọc các cặp ảnh có tỷ lệ thay đổi 
đạt ngưỡng (>25%) để bổ sung vào tập LEVIR, giúp giảm tình trạng lệch class.
"""
import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

def calculate_change_ratio(mask_path):
    # Tính toán tỷ lệ pixel trắng trên tổng diện tích mask
    pass

def filter_cdd_by_threshold(src_paths, dst_paths, threshold=0.25):
    # Lọc và copy các ảnh từ CDD thỏa mãn ngưỡng thay đổi vào thư mục tạm
    pass