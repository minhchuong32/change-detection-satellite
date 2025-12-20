import shutil
import os

def create_archive(source_dir, output_filename):
    """
    Nén toàn bộ thư mục dataset thành định dạng .zip để chuẩn bị cho việc tải xuống hoặc lưu trữ.
    """
    print(f"Archiving {source_dir}...")
    shutil.make_archive(output_filename, 'zip', source_dir)

def clean_working_directory(keep_file_name):
    """
    Dọn dẹp môi trường làm việc bằng cách xóa các thư mục tạm và file thừa, chỉ giữ lại file kết quả cuối cùng.
    """
    working_dir = '/kaggle/working'
    for item in os.listdir(working_dir):
        item_path = os.path.join(working_dir, item)
        if item != keep_file_name:
            if os.path.isfile(item_path): os.remove(item_path)
            elif os.path.isdir(item_path): shutil.rmtree(item_path)