import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

def calculate_change_ratio(mask_path):
    """
    Tính toán tỷ lệ phần trăm các pixel có sự thay đổi (giá trị > 0) trong ảnh mask.
    """
    mask = cv2.imread(mask_path, 0)
    if mask is None: return 0.0
    return np.count_nonzero(mask) / mask.size

def filter_dataset(src_dirs, dst_dirs, threshold=0.25, prefix="cdd_"):
    """
    Lọc các cặp ảnh từ dataset bổ sung dựa trên ngưỡng thay đổi và lưu vào thư mục đích với tiền tố mới.
    """
    for p in dst_dirs.values(): os.makedirs(p, exist_ok=True)
    
    list_files = os.listdir(src_dirs['Label'])
    count = 0
    for filename in tqdm(list_files):
        ratio = calculate_change_ratio(os.path.join(src_dirs['Label'], filename))
        if ratio >= threshold:
            new_name = f"{prefix}{filename}"
            shutil.copy(os.path.join(src_dirs['A'], filename), os.path.join(dst_dirs['A'], new_name))
            shutil.copy(os.path.join(src_dirs['B'], filename), os.path.join(dst_dirs['B'], new_name))
            shutil.copy(os.path.join(src_dirs['Label'], filename), os.path.join(dst_dirs['Label'], new_name))
            count += 1
    return count