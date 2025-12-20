import os
import cv2
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_category(ratio):
    """
    Phân loại mức độ thay đổi dựa trên tỷ lệ diện tích pixel (Low <= 25%, Medium 25-75%, High >= 75%).
    """
    if ratio >= 0.75: return "High"
    elif ratio > 0.25: return "Medium"
    else: return "Low"

def scan_dataset(root_dirs):
    """
    Quét danh sách file từ các thư mục train/test hiện có và tính toán tỷ lệ thay đổi để tạo metadata.
    """
    data = []
    for root in root_dirs:
        label_dir = os.path.join(root, 'label')
        for fname in tqdm(os.listdir(label_dir), desc=f"Scanning {os.path.basename(root)}"):
            mask = cv2.imread(os.path.join(label_dir, fname), 0)
            if mask is None: continue
            ratio = np.count_nonzero(mask) / mask.size
            data.append({'filename': fname, 'src': root, 'cat': get_category(ratio)})
    return pd.DataFrame(data)

def redistribute_to_folders(df, split_name, base_output):
    """
    Thực hiện sao chép các tệp từ metadata vào cấu trúc thư mục train/val/test mới.
    """
    for sub in ['A', 'B', 'label']: os.makedirs(os.path.join(base_output, split_name, sub), exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Moving to {split_name}"):
        for sub in ['A', 'B', 'label']:
            shutil.copy(os.path.join(row['src'], sub, row['filename']), os.path.join(base_output, split_name, sub, row['filename']))

if __name__ == "__main__":
    INPUT_DIRS = ['/kaggle/working/satellite_dataset/train', '/kaggle/working/satellite_dataset/test']
    FINAL_OUTPUT = '/kaggle/working/satellite_full_dataset'
    
    df_all = scan_dataset(INPUT_DIRS)
    
    # Chia 70% Train, 30% Temp (để chia tiếp Val/Test)
    train_df, temp_df = train_test_split(df_all, test_size=0.3, stratify=df_all['cat'], random_state=42)
    # Chia Temp thành Val (20% tổng) và Test (10% tổng)
    val_df, test_df = train_test_split(temp_df, test_size=1/3, stratify=temp_df['cat'], random_state=42)
    
    for df, name in zip([train_df, val_df, test_df], ['train', 'val', 'test']):
        redistribute_to_folders(df, name, FINAL_OUTPUT)