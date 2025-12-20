"""
Công cụ hỗ trợ hiển thị cặp ảnh Before/After cùng với Mask.
Phục vụ mục đích Sanity Check (kiểm tra trực quan) sau mỗi bước xử lý.
"""
import cv2
import matplotlib.pyplot as plt

def visualize_change_pair(path_a, path_b, path_label, title="Sample"):
    # Hiển thị 3 ảnh nằm ngang để so sánh sự thay đổi
    pass