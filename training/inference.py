"""
Inference & Post-processing - Phần của CHƯƠNG
Nhiệm vụ: (7) Inference và hậu xử lý Change Mask
"""

import torch
import cv2
import numpy as np
from typing import Tuple, List


class ChangeDetectionInference:
    """Inference và hậu xử lý cho Change Detection"""
    
    def __init__(self, model, device='cuda', threshold=0.5):
        """
        Args:
            model: Trained U-Net model
            device: 'cuda' hoặc 'cpu'
            threshold: Ngưỡng để tạo binary mask
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self,
               img_before: np.ndarray,
               img_after: np.ndarray,
               apply_postprocess: bool = True) -> np.ndarray:
        """
        Dự đoán change mask
        
        Args:
            img_before: Ảnh trước (H, W) hoặc (H, W, 1)
            img_after: Ảnh sau (H, W) hoặc (H, W, 1)
            apply_postprocess: Có áp dụng post-processing không
        
        Returns:
            Binary change mask (H, W)
        """
        # Chuẩn bị input
        input_tensor = self._prepare_input(img_before, img_after)
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Convert to probability
        prob_mask = torch.sigmoid(output)
        
        # Threshold
        binary_mask = (prob_mask > self.threshold).float()
        
        # Convert to numpy
        binary_mask = binary_mask.cpu().numpy()[0, 0]
        
        # Post-processing
        if apply_postprocess:
            binary_mask = self.post_process(binary_mask)
        
        return binary_mask.astype(np.uint8)
    
    def _prepare_input(self,
                      img_before: np.ndarray,
                      img_after: np.ndarray) -> torch.Tensor:
        """
        Chuẩn bị input cho model
        
        Returns:
            Tensor shape [1, 2, H, W]
        """
        # Ensure grayscale
        if len(img_before.shape) == 3:
            img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        if len(img_after.shape) == 3:
            img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
        
        # Normalize to [0, 1]
        img_before = img_before.astype(np.float32) / 255.0
        img_after = img_after.astype(np.float32) / 255.0
        
        # Stack thành [2, H, W]
        img_pair = np.stack([img_before, img_after], axis=0)
        
        # Add batch dimension [1, 2, H, W]
        img_tensor = torch.from_numpy(img_pair).unsqueeze(0)
        
        return img_tensor
    
    def post_process(self,
                    mask: np.ndarray,
                    min_area: int = 50,
                    kernel_size: int = 5) -> np.ndarray:
        """
        Hậu xử lý change mask
        
        Pipeline:
        1. Morphological closing (lấp lỗ nhỏ)
        2. Remove small objects (loại nhiễu)
        3. Smooth edges
        
        Args:
            mask: Binary mask đầu vào
            min_area: Diện tích tối thiểu (pixels) để giữ lại
            kernel_size: Kích thước kernel cho morphology
        
        Returns:
            Cleaned binary mask
        """
        mask = mask.astype(np.uint8)
        
        # Step 1: Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Step 2: Remove small objects
        cleaned = self._remove_small_regions(closed, min_area)
        
        # Step 3: Smooth edges (optional)
        kernel_smooth = np.ones((3, 3), np.uint8)
        smoothed = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_smooth)
        
        return smoothed
    
    def _remove_small_regions(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """Loại bỏ các vùng nhỏ hơn min_area"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # Create output mask
        output_mask = np.zeros_like(mask)
        
        # Keep only large components (skip label 0 = background)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                output_mask[labels == i] = 1
        
        return output_mask
    
    def extract_change_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Trích xuất đường viền của các vùng thay đổi
        
        Returns:
            List of contours (mỗi contour là array of points)
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
    
    def visualize_changes(self,
                         img_after: np.ndarray,
                         mask: np.ndarray,
                         color: Tuple[int, int, int] = (255, 0, 0),
                         alpha: float = 0.5,
                         draw_contours: bool = True) -> np.ndarray:
        """
        Tạo visualization với mask overlay
        
        Args:
            img_after: Ảnh sau (để vẽ lên)
            mask: Change mask
            color: Màu overlay (B, G, R)
            alpha: Độ trong suốt
            draw_contours: Có vẽ đường viền không
        
        Returns:
            RGB image với visualization
        """
        # Convert to RGB nếu cần
        if len(img_after.shape) == 2:
            img_rgb = cv2.cvtColor(img_after, cv2.COLOR_GRAY2BGR)
        else:
            img_rgb = img_after.copy()
        
        # Tạo colored overlay
        overlay = img_rgb.copy()
        overlay[mask == 1] = color
        
        # Blend
        result = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)
        
        # Draw contours
        if draw_contours:
            contours = self.extract_change_contours(mask)
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        return result


class BatchInference:
    """Inference cho nhiều ảnh"""
    
    def __init__(self, model, device='cuda', batch_size=4):
        self.inference = ChangeDetectionInference(model, device)
        self.batch_size = batch_size
    
    def predict_batch(self,
                     pairs: List[Tuple[str, str]],
                     output_dir: str = './predictions') -> List[np.ndarray]:
        """
        Dự đoán cho nhiều cặp ảnh
        
        Args:
            pairs: List of (before_path, after_path)
            output_dir: Thư mục lưu kết quả
        
        Returns:
            List of predicted masks
        """
        import os
        from PIL import Image
        from tqdm import tqdm
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for i, (before_path, after_path) in enumerate(tqdm(pairs, desc="Predicting")):
            # Load images
            img_before = np.array(Image.open(before_path).convert('L'))
            img_after = np.array(Image.open(after_path).convert('L'))
            
            # Predict
            mask = self.inference.predict(img_before, img_after)
            results.append(mask)
            
            # Save mask
            mask_filename = f"change_mask_{i:04d}.png"
            cv2.imwrite(os.path.join(output_dir, mask_filename), mask * 255)
            
            # Save visualization
            vis = self.inference.visualize_changes(img_after, mask)
            vis_filename = f"visualization_{i:04d}.png"
            cv2.imwrite(os.path.join(output_dir, vis_filename), vis)
        
        return results


# Test script
if __name__ == "__main__":
    from models.unet import UNet
    
    print("🧪 Testing Inference Pipeline...")
    
    # Load model (giả sử đã train)
    model = UNet(n_channels=2, n_classes=1)
    # model.load_state_dict(torch.load('best_model.pth'))
    
    # Tạo inference engine
    inference = ChangeDetectionInference(model, device='cpu', threshold=0.5)
    
    # Tạo dummy data
    np.random.seed(42)
    img_before = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    img_after = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    # Predict
    mask = inference.predict(img_before, img_after, apply_postprocess=True)
    
    print(f"✅ Predicted mask shape: {mask.shape}")
    print(f"✅ Change pixels: {np.sum(mask == 1)}")
    
    # Visualize
    vis = inference.visualize_changes(img_after, mask)
    print(f"✅ Visualization shape: {vis.shape}")
    
    print("\n✅ Inference test completed!")