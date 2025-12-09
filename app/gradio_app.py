"""
Gradio Web App - Phần của CHƯƠNG
Nhiệm vụ: (9) Build Web App với Gradio
"""

import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from typing import Tuple


class ChangeDetectionApp:
    """Web Application cho SAR Change Detection"""
    
    def __init__(self, model_path: str = 'best_model.pth', device: str = 'cpu'):
        """
        Args:
            model_path: Đường dẫn đến file trọng số model
            device: 'cuda' hoặc 'cpu'
        """
        self.device = device
        
        # Load model
        from models.unet import UNet
        self.model = UNet(n_channels=2, n_classes=1)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(device)
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("Using random initialization for demo")
        
        # Inference engine
        from inference import ChangeDetectionInference
        self.inference = ChangeDetectionInference(self.model, device, threshold=0.5)
    
    def process_images(self,
                      img_before: np.ndarray,
                      img_after: np.ndarray,
                      threshold: float = 0.5,
                      min_area: int = 50) -> Tuple:
        """
        Xử lý 2 ảnh và trả về kết quả
        
        Returns:
            (visualization, statistics, histogram)
        """
        # Update threshold
        self.inference.threshold = threshold
        
        # Predict
        mask = self.inference.predict(
            img_before, img_after,
            apply_postprocess=True
        )
        
        # Post-process với min_area tùy chỉnh
        mask = self.inference.post_process(mask, min_area=min_area)
        
        # Visualization
        vis = self.inference.visualize_changes(
            img_after, mask,
            color=(0, 0, 255),  # Red in BGR
            alpha=0.5,
            draw_contours=True
        )
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        
        # Calculate statistics
        stats = self._calculate_statistics(img_before, img_after, mask)
        
        # Create histogram
        hist_plot = self._create_histogram(img_before, img_after)
        
        return vis, stats, hist_plot
    
    def _calculate_statistics(self,
                             img_before: np.ndarray,
                             img_after: np.ndarray,
                             mask: np.ndarray) -> str:
        """Tính toán các thống kê"""
        total_pixels = mask.size
        changed_pixels = np.sum(mask == 1)
        unchanged_pixels = total_pixels - changed_pixels
        
        change_percentage = (changed_pixels / total_pixels) * 100
        
        # Độ chênh lệch trung bình
        diff = np.abs(img_after.astype(float) - img_before.astype(float))
        mean_change = np.mean(diff[mask == 1]) if changed_pixels > 0 else 0
        
        # Number of change regions
        num_regions = len(self.inference.extract_change_contours(mask))
        
        stats_text = f"""
## 📊 CHANGE DETECTION STATISTICS

### Overall Metrics
- **Total Pixels**: {total_pixels:,}
- **Changed Pixels**: {changed_pixels:,}
- **Unchanged Pixels**: {unchanged_pixels:,}
- **Change Percentage**: {change_percentage:.2f}%

### Change Characteristics
- **Number of Change Regions**: {num_regions}
- **Mean Intensity Change**: {mean_change:.2f}
- **Largest Region Area**: {self._get_largest_region_area(mask):,} pixels

### Image Properties
- **Image Size**: {img_before.shape[0]} × {img_before.shape[1]} pixels
- **Before Image Mean**: {np.mean(img_before):.2f}
- **After Image Mean**: {np.mean(img_after):.2f}
"""
        return stats_text
    
    def _get_largest_region_area(self, mask: np.ndarray) -> int:
        """Tìm diện tích vùng lớn nhất"""
        contours = self.inference.extract_change_contours(mask)
        if not contours:
            return 0
        areas = [cv2.contourArea(c) for c in contours]
        return int(max(areas)) if areas else 0
    
    def _create_histogram(self,
                         img_before: np.ndarray,
                         img_after: np.ndarray) -> go.Figure:
        """Tạo histogram so sánh 2 ảnh"""
        # Flatten images
        before_flat = img_before.flatten()
        after_flat = img_after.flatten()
        
        # Create figure
        fig = go.Figure()
        
        # Before histogram
        fig.add_trace(go.Histogram(
            x=before_flat,
            name='Before',
            opacity=0.7,
            marker_color='blue',
            nbinsx=50
        ))
        
        # After histogram
        fig.add_trace(go.Histogram(
            x=after_flat,
            name='After',
            opacity=0.7,
            marker_color='red',
            nbinsx=50
        ))
        
        fig.update_layout(
            title='Intensity Distribution Comparison',
            xaxis_title='Pixel Intensity',
            yaxis_title='Frequency',
            barmode='overlay',
            template='plotly_white'
        )
        
        return fig


def create_demo():
    """Tạo Gradio interface"""
    
    # Initialize app
    app = ChangeDetectionApp(model_path='best_model.pth', device='cpu')
    
    # Create interface
    with gr.Blocks(title="SAR Change Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🛰️ SAR Image Change Detection
        
        Upload two SAR images (Before and After) to detect changes between them.
        The system uses a U-Net deep learning model trained on SAR imagery.
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📥 Input Images")
                img_before = gr.Image(label="Before Image", type="numpy")
                img_after = gr.Image(label="After Image", type="numpy")
                
                with gr.Row():
                    threshold_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        label="Detection Threshold"
                    )
                    min_area_slider = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=50,
                        step=10,
                        label="Minimum Change Area (pixels)"
                    )
                
                detect_btn = gr.Button("🔍 Detect Changes", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### 📤 Results")
                output_image = gr.Image(label="Change Detection Result")
                
        with gr.Row():
            with gr.Column():
                stats_output = gr.Markdown(label="Statistics")
            
            with gr.Column():
                hist_output = gr.Plot(label="Intensity Histogram")
        
        # Examples
        gr.Markdown("### 📚 Examples")
        gr.Examples(
            examples=[
                ["examples/before_1.tif", "examples/after_1.tif", 0.5, 50],
                ["examples/before_2.tif", "examples/after_2.tif", 0.4, 30],
            ],
            inputs=[img_before, img_after, threshold_slider, min_area_slider],
        )
        
        # Event handler
        detect_btn.click(
            fn=app.process_images,
            inputs=[img_before, img_after, threshold_slider, min_area_slider],
            outputs=[output_image, stats_output, hist_output]
        )
        
        gr.Markdown("""
        ---
        ### ℹ️ About
        
        **Change Detection Methods:**
        - Deep Learning: U-Net architecture with encoder-decoder
        - Pre-processing: Speckle filtering (Lee/Frost/Median)
        - Post-processing: Morphological operations, small object removal
        
        **Visualization:**
        - Red overlay: Detected change regions
        - Green contours: Boundaries of change regions
        
        **Dataset:** SAR imagery from multiple seasons (Spring → Winter)
        """)
    
    return demo


# Run app
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )