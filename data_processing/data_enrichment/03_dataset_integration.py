import os
import shutil
from tqdm import tqdm

def copy_with_prefix(src_dir, dst_dir, prefix="", description=""):
    """
    Sao chép tệp từ thư mục nguồn sang đích, hỗ trợ thêm tiền tố (prefix) để tránh trùng tên và hiển thị tiến trình.
    """
    if not os.path.exists(src_dir):
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    files = os.listdir(src_dir)
    for filename in tqdm(files, desc=description, leave=False):
        shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_dir, f"{prefix}{filename}"))
    return len(files)

def merge_split_data(split_name, levir_base, supp_base, final_base):
    """
    Hàm dùng chung để gộp dữ liệu từ nguồn LEVIR và Supplement (CDD) vào một thư mục đích cụ thể (train hoặc test).
    """
    print(f"Đang gộp dữ liệu cho tập: {split_name.upper()}")
    dest_path = os.path.join(final_base, split_name)
    
    sources = [
        (os.path.join(levir_base, split_name), ""), 
        (supp_base, "cdd_")
    ]
    
    for src_root, prefix in sources:
        for sub in ['A', 'B', 'label' if 'label' in os.listdir(src_root) else 'OUT']:
            dst_sub = 'label' if sub in ['label', 'OUT'] else sub
            copy_with_prefix(os.path.join(src_root, sub), os.path.join(dest_path, dst_sub), prefix, f"{split_name}-{sub}")

if __name__ == "__main__":
    FINAL_WORKING_DIR = '/kaggle/working/satellite_dataset'
    
    # Gộp tập TRAIN
    merge_split_data(
        split_name='train',
        levir_base='/kaggle/input/levir-processed/kaggle/working/processed_balanced_levircd',
        supp_base='/kaggle/working/supplement_data', # Lấy từ bước filter CDD train
        final_base=FINAL_WORKING_DIR
    )
    
    # Gộp tập TEST
    merge_split_data(
        split_name='test',
        levir_base='/kaggle/input/levir-processed/kaggle/working/processed_balanced_levircd',
        supp_base='/kaggle/working/cdd_test_filtered', # Lấy từ bước filter CDD test
        final_base=FINAL_WORKING_DIR
    )