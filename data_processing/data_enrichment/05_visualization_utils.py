import os
import cv2
import matplotlib.pyplot as plt

def visualize_samples(base_path, num_samples=5):
    """
    Hiển thị ngẫu nhiên các bộ ảnh (Before, After, Label) để kiểm tra trực quan chất lượng dữ liệu.
    """
    files = sorted(os.listdir(os.path.join(base_path, 'A')))[:num_samples]
    plt.figure(figsize=(15, 5 * num_samples))
    for i, fname in enumerate(files):
        # Đọc và chuyển đổi hệ màu BGR sang RGB để hiển thị bằng Matplotlib
        pass
    plt.tight_layout()
    plt.show()