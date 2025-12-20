import os
import shutil

def archive_dataset(source_path, output_zip_name):
    """
    Nén toàn bộ thư mục dataset thành file .zip để tải xuống hoặc lưu trữ trên Kaggle Output.
    """
    print(f"📦 Archiving {source_path}...")
    shutil.make_archive(output_zip_name, 'zip', source_path)
    return f"{output_zip_name}.zip"

def clean_workspace(keep_file_name):
    """
    Xóa tất cả các file và thư mục tạm trong /kaggle/working, chỉ giữ lại file zip kết quả cuối cùng.
    """
    working_dir = '/kaggle/working'
    print(f"Cleaning {working_dir}...")
    for item in os.listdir(working_dir):
        item_path = os.path.join(working_dir, item)
        if item == os.path.basename(keep_file_name): continue
        if os.path.isfile(item_path) or os.path.islink(item_path): os.remove(item_path)
        elif os.path.isdir(item_path): shutil.rmtree(item_path)

if __name__ == "__main__":
    DATA_PATH = '/kaggle/working/satellite_full_dataset'
    ZIP_NAME = '/kaggle/working/satellite_change_detection_final'
    
    zip_res = archive_dataset(DATA_PATH, ZIP_NAME)
    clean_workspace(zip_res)
    print(f"DONE! Final file: {zip_res}")