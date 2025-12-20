"""
Kiểm tra tính đồng bộ giữa các thư mục A, B, Label của tập LEVIR. 
Đồng thời thống kê tỷ lệ pixel thay đổi (Change Ratio) để đánh giá độ cân bằng của Dataset.
"""
import os
import cv2
import numpy as np
from termcolor import colored
from tqdm import tqdm

def get_clean_file_list(folder_path):
    # Lấy danh sách file ảnh hợp lệ,
    pass

def check_consistency(paths):
    # Kiểm tra sự khớp nhau về tên file và số lượng giữa A, B và Label
    pass

def analyze_change_distribution(label_dir):
    # Thống kê tỷ lệ diện tích thay đổi (Low <= 25%, Medium, High>=75%) trong tập dữ liệu
    pass