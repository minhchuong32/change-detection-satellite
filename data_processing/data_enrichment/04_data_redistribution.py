import os
import cv2
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_category(ratio):
    """
    Phân loại mức độ thay đổi của ảnh thành 3 nhóm: Low, Medium, High dựa trên tỷ lệ pixel.
    """
    if ratio >= 0.75: return "High"
    elif ratio > 0.25: return "Medium"
    return "Low"

def scan_and_build_metadata(root_dir):
    """
    Quét thư mục dataset để tạo DataFrame chứa thông tin về đường dẫn, tỷ lệ thay đổi và nhóm phân loại.
    """
    data = []
    label_dir = os.path.join(root_dir, 'label')
    for filename in tqdm(os.listdir(label_dir)):
        mask = cv2.imread(os.path.join(label_dir, filename), 0)
        if mask is None: continue
        ratio = np.count_nonzero(mask) / mask.size
        data.append({'filename': filename, 'src': root_dir, 'cat': get_category(ratio)})
    return pd.DataFrame(data)

def split_and_move(df, output_base, train_size=0.7, val_size=0.2, test_size=0.1):
    """
    Thực hiện Stratified Split dựa trên nhóm phân loại và di chuyển file vào các thư mục train/val/test tương ứng.
    """
    # Logic sử dụng train_test_split và shutil.copy để chia lại dataset
    pass