import os
import numpy as np
from PIL import Image
from pathlib import Path

# --- Cấu hình Dữ liệu Demo ---
TARGET_DATA_ROOT = "data"
# Chỉ tạo 2 mùa để test chức năng ghép cặp (Spring -> Winter)
SEASONS_TO_CREATE = ["spring", "winter"] 
SCENES_TO_CREATE = ["s_01", "s_02"] # 2 vùng khảo sát
PATCHES_TO_CREATE = ["p_001", "p_002"] # 2 patch trong mỗi vùng
IMAGE_SIZE = (256, 256)
DTYPE = np.uint16 # Kiểu dữ liệu 16-bit cho ảnh SAR

def create_dummy_image(size: tuple, dtype: np.dtype, unique_id: str) -> np.ndarray:
    """Tạo mảng numpy giả lập ảnh SAR 16-bit."""
    H, W = size
    
    # Tạo giá trị ngẫu nhiên, tập trung vào dải giữa của 16-bit (0-65535)
    img = np.random.randint(2000, 6000, size=(H, W), dtype=dtype)
    
    # Thêm một chút khác biệt dựa trên ID để mô phỏng "thay đổi"
    if 'spring' in unique_id:
        img[50:100, 50:150] = np.clip(img[50:100, 50:150] + 1000, 0, 65535)
    if 'winter' in unique_id:
        img[150:200, 150:200] = np.clip(img[150:200, 150:200] - 1000, 0, 65535)

    # Thêm một đường chéo nhỏ để đảm bảo ảnh không hoàn toàn ngẫu nhiên
    np.fill_diagonal(img, 10000)
    
    return img

def save_image_as_tif(np_array: np.ndarray, path: Path):
    """Lưu mảng numpy 16-bit dưới dạng TIFF sử dụng PIL."""
    # Chuyển numpy array sang đối tượng Image của PIL
    img_pil = Image.fromarray(np_array)
    
    # Lưu với định dạng TIFF. PIL sẽ tự động xử lý độ sâu 16-bit
    img_pil.save(path)

def create_demo_data(target_root: str):
    """Tạo cấu trúc thư mục và các tệp ảnh demo."""
    target_root_path = Path(target_root)
    print(f"Bắt đầu tạo dữ liệu demo tại: {target_root_path.resolve()}")
    
    for season in SEASONS_TO_CREATE:
        for s_id in SCENES_TO_CREATE:
            for p_id in PATCHES_TO_CREATE:
                # 1. Định nghĩa đường dẫn đích
                # data/spring/s_01/p_001.tif
                scene_dir = target_root_path / season / s_id
                scene_dir.mkdir(parents=True, exist_ok=True)
                
                file_name = f"{p_id}.tif"
                image_path = scene_dir / file_name
                
                # 2. Tạo ảnh numpy giả lập
                unique_id = f"{season}_{s_id}_{p_id}"
                dummy_img = create_dummy_image(IMAGE_SIZE, DTYPE, unique_id)
                
                # 3. Lưu ảnh dưới dạng TIFF
                save_image_as_tif(dummy_img, image_path)
                
                print(f"Đã tạo file: {image_path.relative_to(target_root_path.parent)}")

    print("\n✅ Hoàn thành tạo dữ liệu demo!")
    print(f"Bạn có thể chạy `SARDatasetLoader` ngay bây giờ.")

if __name__ == "__main__":
    create_demo_data(TARGET_DATA_ROOT)
    
    # Sau khi tạo xong, chạy thử nghiệm lớp SARDatasetLoader của bạn
    print("\n--- Chạy thử nghiệm SARDatasetLoader ---")
    
    # Import lại lớp của bạn để kiểm tra
    # Lưu ý: Giả định code của bạn đã được lưu trong file khác (ví dụ: dataset_loader.py)
    # Nếu không, hãy copy lại class SARDatasetLoader vào đây để chạy.
    
    try:
        loader = SARDatasetLoader(data_root=TARGET_DATA_ROOT)
        dataset = loader.load_dataset()
        
        # Kiểm tra truy cập
        spring_s01_p001_path = loader.get_image_path('spring', 's_01', 'p_001')
        print(f"\n📍 Ví dụ đường dẫn: {spring_s01_p001_path}")
        
    except NameError:
        print("\n⚠️ Vui lòng đảm bảo class SARDatasetLoader đã được import hoặc định nghĩa lại.")