"""
Mô tả: Thực hiện gộp các nguồn dữ liệu (LEVIR gốc + CCD đã lọc). 
Tự động thêm prefix ('cdd_') để tránh trùng tên file khi trộn.
"""
import os
import shutil
from tqdm import tqdm

def merge_sources(sources, destination, prefix=""):
    pass