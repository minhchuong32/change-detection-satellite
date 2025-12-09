"""
Ground Truth Generator - Phần của MINH
Nhiệm vụ: (4) Tạo Ground Truth Change Map
"""

import numpy as np
import cv2
from skimage import morphology
from skimage.filters import threshold_otsu
from typing import Tuple


class GroundTruthGenerator:
    """
    Tạo Ground Truth Change Map từ cặp ảnh Before-After
    Vì dataset không có label → Tự động tạo bằng phương pháp Image Differencing
    """
    
    def __init__(self, min_change_area: int = 50):
        """
        Args:
            min_change_area: Diện tích tối thiểu (pixels) để coi là thay đổi thật
        """
        self.min_change_area = min_change_area
    
    def generate_change_mask(self, 
                           img_before: np.ndarray,
                           img_after: np.ndarray,
                           method: str = 'otsu') -> np.ndarray:
        """
        Tạo change mask từ 2 ảnh
        
        Pipeline:
        1. Tính ảnh chênh lệch tuyệt đối: D = |After - Before|
        2. Áp dụng threshold (Otsu hoặc manual)
        3. Morphological closing để lấp lỗ hổng
        4. Loại bỏ vùng nhỏ (noise)
        
        Args:
            img_before: Ảnh trước (grayscale)
            img_after: Ảnh sau (grayscale)
            method: 'otsu', 'mean', hoặc 'adaptive'
        
        Returns:
            Binary mask (0: không đổi, 1: thay đổi)
        """
        # Bước 1: Tính ảnh chênh lệch tuyệt đối
        diff_img = self._compute_difference(img_before, img_after)
        
        # Bước 2: Thresholding
        if method == 'otsu':
            binary_mask = self._threshold_otsu(diff_img)
        elif method == 'mean':
            binary_mask = self._threshold_mean(diff_img)
        elif method == 'adaptive':
            binary_mask = self._threshold_adaptive(diff_img)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Bước 3: Morphological closing
        binary_mask = self._morphological_closing(binary_mask)
        
        # Bước 4: Remove small objects
        binary_mask = self._remove_small_objects(binary_mask)
        
        return binary_mask
    
    def _compute_difference(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Tính ảnh chênh lệch
        
        Công thức: D = |I2 - I1|
        """
        # Chuyển về float để tính toán chính xác
        img1_f = img1.astype(np.float32)
        img2_f = img2.astype(np.float32)
        
        # Absolute difference
        diff = np.abs(img2_f - img1_f)
        
        return diff.astype(np.uint8)
    
    def _threshold_otsu(self, diff_img: np.ndarray) -> np.ndarray:
        """
        Otsu's threshold - Tự động tìm ngưỡng tối ưu
        
        Otsu tìm threshold sao cho phương sai giữa 2 class (đổi/không đổi) là lớn nhất
        """
        try:
            thresh_val = threshold_otsu(diff_img)
            print(f"  📊 Otsu threshold: {thresh_val:.2f}")
        except:
            # Fallback nếu Otsu fail
            thresh_val = np.mean(diff_img) + np.std(diff_img)
            print(f"  ⚠️ Otsu failed, using mean+std: {thresh_val:.2f}")
        
        binary_mask = (diff_img > thresh_val).astype(np.uint8)
        return binary_mask
    
    def _threshold_mean(self, diff_img: np.ndarray) -> np.ndarray:
        """Threshold dựa trên mean + k*std"""
        thresh_val = np.mean(diff_img) + 2 * np.std(diff_img)
        print(f"  📊 Mean+2*std threshold: {thresh_val:.2f}")
        binary_mask = (diff_img > thresh_val).astype(np.uint8)
        return binary_mask
    
    def _threshold_adaptive(self, diff_img: np.ndarray) -> np.ndarray:
        """Adaptive threshold - Tốt cho ảnh có độ sáng không đồng đều"""
        binary_mask = cv2.adaptiveThreshold(
            diff_img,
            maxValue=1,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        return binary_mask
    
    def _morphological_closing(self, binary_mask: np.ndarray, 
                               kernel_size: int = 5) -> np.ndarray:
        """
        Morphological closing: Dilation + Erosion
        Mục đích: Lấp các lỗ nhỏ trong vùng thay đổi
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        return closed_mask
    
    def _remove_small_objects(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Loại bỏ các vùng nhỏ (nhiễu)
        """
        # Chuyển sang boolean
        mask_bool = binary_mask.astype(bool)
        
        # Remove small objects
        cleaned_mask = morphology.remove_small_objects(
            mask_bool,
            min_size=self.min_change_area
        )
        
        return cleaned_mask.astype(np.uint8)
    
    def compute_statistics(self, 
                          img_before: np.ndarray,
                          img_after: np.ndarray,
                          mask: np.ndarray) -> dict:
        """
        Tính các thống kê về vùng thay đổi
        
        Returns:
            Dictionary chứa các metrics
        """
        total_pixels = mask.size
        changed_pixels = np.sum(mask == 1)
        unchanged_pixels = total_pixels - changed_pixels
        
        change_percentage = (changed_pixels / total_pixels) * 100
        
        # Độ chênh lệch trung bình trong vùng thay đổi
        diff = np.abs(img_after.astype(float) - img_before.astype(float))
        mean_change_intensity = np.mean(diff[mask == 1]) if changed_pixels > 0 else 0
        
        stats = {
            'total_pixels': total_pixels,
            'changed_pixels': int(changed_pixels),
            'unchanged_pixels': int(unchanged_pixels),
            'change_percentage': float(change_percentage),
            'mean_change_intensity': float(mean_change_intensity)
        }
        
        return stats
    
    def visualize_change_detection(self,
                                   img_before: np.ndarray,
                                   img_after: np.ndarray,
                                   mask: np.ndarray) -> np.ndarray:
        """
        Tạo ảnh visualization: After + Mask overlay (màu đỏ)
        
        Returns:
            RGB image với vùng thay đổi được highlight màu đỏ
        """
        # Chuyển ảnh after sang RGB
        if len(img_after.shape) == 2:
            img_rgb = cv2.cvtColor(img_after, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_after.copy()
        
        # Tạo overlay màu đỏ cho vùng thay đổi
        red_overlay = np.zeros_like(img_rgb)
        red_overlay[:, :, 2] = 255  # Red channel
        
        # Blend với alpha
        alpha = 0.5
        img_rgb[mask == 1] = cv2.addWeighted(
            img_rgb[mask == 1], 1 - alpha,
            red_overlay[mask == 1], alpha,
            0
        )
        
        return img_rgb


# Test script
if __name__ == "__main__":
    print("🧪 Testing Ground Truth Generator...")
    
    # Tạo 2 ảnh test
    np.random.seed(42)
    h, w = 256, 256
    
    # Before: background đồng nhất
    img_before = np.ones((h, w), dtype=np.uint8) * 100
    
    # After: thêm một vùng thay đổi
    img_after = img_before.copy()
    img_after[80:150, 80:180] = 200  # Vùng sáng hơn
    
    # Thêm noise
    img_before += np.random.randint(-10, 10, (h, w)).astype(np.uint8)
    img_after += np.random.randint(-10, 10, (h, w)).astype(np.uint8)
    
    # Tạo GT generator
    gt_gen = GroundTruthGenerator(min_change_area=50)
    
    # Test các phương pháp
    print("\n📊 Testing different methods:")
    mask_otsu = gt_gen.generate_change_mask(img_before, img_after, method='otsu')
    mask_mean = gt_gen.generate_change_mask(img_before, img_after, method='mean')
    
    # Statistics
    stats = gt_gen.compute_statistics(img_before, img_after, mask_otsu)
    print("\n📈 Change Statistics (Otsu method):")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Test completed!")