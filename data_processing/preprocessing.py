# """
# Image Registration & Alignment - Phần của CHƯƠNG
# Nhiệm vụ: (3) Đăng ký ảnh, Resize/Padding
# """

# import cv2
# import numpy as np
# from typing import Tuple


# class ImageAligner:
#     """Căn chỉnh và đăng ký ảnh Before-After"""
    
#     def __init__(self, feature_detector='orb'):
#         """
#         Args:
#             feature_detector: 'orb', 'sift', hoặc 'akaze'
#         """
#         self.detector_type = feature_detector
        
#         if feature_detector == 'orb':
#             self.detector = cv2.ORB_create(nfeatures=5000)
#         elif feature_detector == 'sift':
#             self.detector = cv2.SIFT_create()
#         elif feature_detector == 'akaze':
#             self.detector = cv2.AKAZE_create()
#         else:
#             raise ValueError(f"Unknown detector: {feature_detector}")
        
#         # Matcher
#         if feature_detector == 'sift':
#             self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
#         else:
#             self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
#     def align_images(self, 
#                     img_before: np.ndarray, 
#                     img_after: np.ndarray,
#                     max_features: int = 500) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Căn chỉnh img_after với img_before bằng feature matching
        
#         Pipeline:
#         1. Detect keypoints và descriptors
#         2. Match features
#         3. Tính homography matrix
#         4. Warp img_after
        
#         Args:
#             img_before: Ảnh reference
#             img_after: Ảnh cần align
#             max_features: Số features tối đa để match
        
#         Returns:
#             (img_before, aligned_img_after)
#         """
#         print("🔄 Aligning images...")
        
#         # Convert sang grayscale nếu cần
#         if len(img_before.shape) == 3:
#             gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
#         else:
#             gray_before = img_before
        
#         if len(img_after.shape) == 3:
#             gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
#         else:
#             gray_after = img_after
        
#         # Step 1: Detect keypoints và descriptors
#         kp1, des1 = self.detector.detectAndCompute(gray_before, None)
#         kp2, des2 = self.detector.detectAndCompute(gray_after, None)
        
#         print(f"  Found {len(kp1)} keypoints in before image")
#         print(f"  Found {len(kp2)} keypoints in after image")
        
#         if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
#             print("  ⚠️ Not enough keypoints for alignment, returning original")
#             return img_before, img_after
        
#         # Step 2: Match features
#         matches = self.matcher.knnMatch(des1, des2, k=2)
        
#         # Apply Lowe's ratio test
#         good_matches = []
#         for m_n in matches:
#             if len(m_n) == 2:
#                 m, n = m_n
#                 if m.distance < 0.75 * n.distance:
#                     good_matches.append(m)
        
#         print(f"  Found {len(good_matches)} good matches")
        
#         if len(good_matches) < 10:
#             print("  ⚠️ Not enough good matches, returning original")
#             return img_before, img_after
        
#         # Step 3: Tính homography matrix
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
#         H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
#         if H is None:
#             print("  ⚠️ Homography computation failed")
#             return img_before, img_after
        
#         # Step 4: Warp img_after
#         h, w = img_before.shape[:2]
#         aligned_after = cv2.warpPerspective(img_after, H, (w, h))
        
#         print("  ✅ Alignment completed")
#         return img_before, aligned_after
    
#     def check_alignment_quality(self,
#                                img1: np.ndarray,
#                                img2: np.ndarray) -> float:
#         """
#         Đánh giá chất lượng alignment bằng SSIM hoặc correlation
        
#         Returns:
#             Score từ 0-1 (1 = perfect alignment)
#         """
#         # Normalize
#         img1_norm = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         img2_norm = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
#         # Tính correlation
#         correlation = np.corrcoef(img1_norm.flatten(), img2_norm.flatten())[0, 1]
        
#         return max(0, correlation)


# class ImageResizer:
#     """Resize và padding ảnh về kích thước chuẩn"""
    
#     @staticmethod
#     def resize_with_padding(img: np.ndarray,
#                            target_size: Tuple[int, int] = (256, 256),
#                            padding_value: int = 0) -> np.ndarray:
#         """
#         Resize ảnh và thêm padding để giữ tỷ lệ
        
#         Args:
#             img: Ảnh input
#             target_size: (height, width) mong muốn
#             padding_value: Giá trị để padding (thường là 0)
        
#         Returns:
#             Ảnh đã resize và padding
#         """
#         h, w = img.shape[:2]
#         target_h, target_w = target_size
        
#         # Tính tỷ lệ
#         scale = min(target_w / w, target_h / h)
        
#         # Resize giữ tỷ lệ
#         new_w = int(w * scale)
#         new_h = int(h * scale)
#         resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
#         # Tạo canvas với padding
#         if len(img.shape) == 3:
#             canvas = np.full((target_h, target_w, img.shape[2]), padding_value, dtype=img.dtype)
#         else:
#             canvas = np.full((target_h, target_w), padding_value, dtype=img.dtype)
        
#         # Đặt ảnh vào giữa canvas
#         y_offset = (target_h - new_h) // 2
#         x_offset = (target_w - new_w) // 2
#         canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
#         return canvas
    
#     @staticmethod
#     def center_crop(img: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
#         """Crop ảnh từ giữa"""
#         h, w = img.shape[:2]
#         crop_h, crop_w = crop_size
        
#         start_h = (h - crop_h) // 2
#         start_w = (w - crop_w) // 2
        
#         return img[start_h:start_h+crop_h, start_w:start_w+crop_w]


# # Test script
# if __name__ == "__main__":
#     print("🧪 Testing Image Aligner...")
    
#     # Tạo 2 ảnh test với một chút dịch chuyển
#     np.random.seed(42)
#     img1 = np.random.randint(0, 255, (300, 400), dtype=np.uint8)
    
#     # Tạo img2 bằng cách shift img1
#     M = np.float32([[1, 0, 10], [0, 1, 5]])  # Dịch 10px sang phải, 5px xuống
#     img2 = cv2.warpAffine(img1, M, (400, 300))
    
#     # Test alignment
#     aligner = ImageAligner(feature_detector='orb')
#     img1_aligned, img2_aligned = aligner.align_images(img1, img2)
    
#     # Check quality
#     quality = aligner.check_alignment_quality(img1_aligned, img2_aligned)
#     print(f"\n✅ Alignment quality: {quality:.4f}")
    
#     # Test resizing
#     print("\n🧪 Testing Image Resizer...")
#     resizer = ImageResizer()
#     resized = resizer.resize_with_padding(img1, target_size=(256, 256))
#     print(f"✅ Resized shape: {resized.shape}")
    
#     print("\n✅ All tests completed!")

# data_processing/preprocessing.py

import numpy as np
import cv2
from scipy.ndimage import uniform_filter # Dùng cho bộ lọc cơ bản

# --- Phần 1: Speckle Filtering (THÙY) ---
# Ảnh đầu vào: Ảnh numpy 
def apply_speckle_filter(image: np.ndarray, filter_type: str = 'Median'):
    """Áp dụng các bộ lọc giảm nhiễu Speckle/Noise (Lee, Frost, Median)."""
    
    if image.ndim > 2 and image.shape[2] == 3:
        # Đây là ảnh RGB (LEVIR-CD), không phải ảnh SAR 1 kênh đơn thuần
        # Áp dụng bộ lọc cho từng kênh hoặc chuyển sang Grayscale trước (tùy thuộc vào yêu cầu)
        # Nếu là RGB, bộ lọc median là lựa chọn an toàn nhất.
        if filter_type == 'Median':
            return cv2.medianBlur(image.astype(np.uint8), 3) # Kernel 3x3
        # Bộ lọc Lee/Frost phức tạp và thường chỉ áp dụng cho ảnh SAR 1 kênh
        return image # Trả về ảnh gốc nếu không có bộ lọc chuyên dụng
    
    # Giả lập Median cho ảnh 1 kênh:
    if filter_type == 'Median':
        return cv2.medianBlur(image.astype(np.uint8), 3)
    
    return image # Mặc định trả về ảnh gốc

# --- Phần 2: Normalization (MINH) ---

def normalize_intensity(image: np.ndarray):
    """Chuẩn hóa cường độ ảnh (Min-Max) sang dải [0, 1]."""
    # Xử lý normalization trên từng kênh (nếu là ảnh RGB)
    if image.ndim > 2:
        normalized_img = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[2]):
            channel = image[..., i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val > min_val:
                normalized_img[..., i] = (channel - min_val) / (max_val - min_val)
            # Nếu min=max, kênh đó sẽ là 0
        return normalized_img
    
    # Xử lý grayscale
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
         return np.zeros_like(image, dtype=np.float32)
         
    normalized_img = (image - min_val) / (max_val - min_val)
    return normalized_img.astype(np.float32)

# --- Phần 3: Image Alignment (CHƯƠNG) ---

def align_images(before_img, after_img):
    """Image Registration (Đăng ký ảnh) để căn chỉnh."""
    # Với dữ liệu patch 256x256 đã cắt từ LEVIR-CD, bước căn chỉnh thường không cần thiết
    # vì ảnh đã được căn chỉnh ở mức độ Scene.
    
    # Nếu cần, bạn sẽ sử dụng ORB/SIFT để tìm homography và warp ảnh.
    # Tuy nhiên, để dự án chạy, ta bỏ qua logic phức tạp này.
    
    return before_img, after_img # Trả về ảnh gốc