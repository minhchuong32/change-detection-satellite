import os
import shutil
from tqdm import tqdm

def copy_with_metadata(src_dir, dst_dir, prefix="", desc=""):
    """
    Hàm bổ trợ để copy toàn bộ nội dung thư mục nguồn sang thư mục đích, hỗ trợ thêm tiền tố cho file.
    """
    os.makedirs(dst_dir, exist_ok=True)
    files = os.listdir(src_dir)
    for filename in tqdm(files, desc=desc):
        shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_dir, f"{prefix}{filename}"))
    return len(files)

def merge_datasets(source_configs, final_base_dir):
    """
    Gộp nhiều nguồn dữ liệu (như LEVIR và CDD đã lọc) vào một cấu trúc thư mục tập trung duy nhất.
    """
    # Logic gộp dữ liệu A, B, Label từ các nguồn khác nhau vào final_base_dir
    pass