import cv2
import numpy as np

def clean_prediction_mask(prediction, threshold=0.5, min_area=50):
    """
    Xử lý hậu kỳ: Khử nhiễu speckle và lọc diện tích vùng thay đổi.
    """
    # 1. Ngưỡng nhị phân
    binary_mask = (prediction > threshold).astype(np.uint8) * 255

    # 2. Phép toán hình thái học (Morphology)
    # Dùng kernel 3x3 để đóng các lỗ hổng nhỏ và xóa nhiễu trắng li ti
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

    # 3. Lọc các vùng có diện tích quá nhỏ (Connected Components)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed, connectivity=8)
    
    final_mask = np.zeros_like(processed)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            final_mask[labels == i] = 1 # Trả về mask 0-1 để dùng làm index
            
    return final_mask

def create_visual_overlay(image_after, binary_mask, alpha=0.3):
    """
    Tạo ảnh lớp phủ (overlay) màu đỏ lên các vùng thay đổi.
    """
    img_np = np.array(image_after)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    overlay = img_np.copy()
    overlay[binary_mask == 1] = [255, 0, 0] # Màu đỏ cho vùng thay đổi
    
    # Trộn ảnh gốc với lớp phủ màu đỏ
    output = cv2.addWeighted(img_np, 1 - alpha, overlay, alpha, 0)
    return output