import gradio as gr
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import timm
import os
import traceback
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.exposure import match_histograms

# ==================== 1. CONFIG & UTILS ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 256

# ==================== 2. MODEL ARCHITECTURES ====================
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

# --- A. Siamese Pure ---
class Encoder_Thuy(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2, self.c3 = DoubleConv(3, 64), DoubleConv(64, 128), DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        f1 = self.c1(x); f2 = self.c2(self.pool(f1)); f3 = self.c3(self.pool(f2))
        return [f1, f2, f3]

class Decoder_Thuy(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c2 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)
    def forward(self, f):
        f1, f2, f3 = f
        x = self.c1(torch.cat([self.up1(f3), f2], 1))
        x = self.c2(torch.cat([self.up2(x), f1], 1))
        return self.out(x)

class SiamesePure(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec = Encoder_Thuy(), Decoder_Thuy()
    def forward(self, a, b):
        fa, fb = self.enc(a), self.enc(b)
        return self.dec([torch.abs(x - y) for x, y in zip(fa, fb)])

# --- B. Siamese MobileNetV2 ---
class SiameseUnetMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=None) 
        self.base_layers = base.features
        self.up1 = nn.ConvTranspose2d(1280, 96, 2, stride=2)
        self.conv1 = ConvBlock(96 + 96, 96)
        self.up2 = nn.ConvTranspose2d(96, 32, 2, stride=2)
        self.conv2 = ConvBlock(32 + 32, 32)
        self.up3 = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.conv3 = ConvBlock(24 + 24, 24)
        self.up4 = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.conv4 = ConvBlock(16 + 16, 16)
        self.final_up = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.final_conv = nn.Conv2d(16, 1, 1)

    def forward_one(self, x):
        x1 = self.base_layers[:2](x); x2 = self.base_layers[2:4](x1)
        x3 = self.base_layers[4:7](x2); x4 = self.base_layers[7:14](x3)
        x5 = self.base_layers[14:](x4)
        return [x1, x2, x3, x4, x5]

    def forward(self, imgA, imgB):
        fA = self.forward_one(imgA); fB = self.forward_one(imgB)
        d = [torch.abs(fA[i] - fB[i]) for i in range(5)]
        u1 = self.up1(d[4]); c1 = self.conv1(torch.cat([u1, d[3]], dim=1))
        u2 = self.up2(c1); c2 = self.conv2(torch.cat([u2, d[2]], dim=1))
        u3 = self.up3(c2); c3 = self.conv3(torch.cat([u3, d[1]], dim=1))
        u4 = self.up4(c3); c4 = self.conv4(torch.cat([u4, d[0]], dim=1))
        return self.final_conv(self.final_up(c4))

# --- C. EfficientNet U-Net ---
class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, features_only=True, output_stride=32)
    def forward(self, x):
        f = self.backbone(x)
        return [f[1], f[2], f[3], f[4]] 

class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [24, 40, 112, 320]
        self.bottleneck_conv = DoubleConv(ch[3], 256)
        self.up1, self.c1 = nn.ConvTranspose2d(256, 128, 2, 2), DoubleConv(128 + ch[2], 128)
        self.up2, self.c2 = nn.ConvTranspose2d(128, 64, 2, 2), DoubleConv(64 + ch[1], 64)
        self.up3, self.c3 = nn.ConvTranspose2d(64, 32, 2, 2), DoubleConv(32 + ch[0], 32)
        self.up4 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.up5 = nn.ConvTranspose2d(16, 16, 2, 2)
        self.c4 = DoubleConv(16, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, f):
        f1, f2, f3, bottleneck = f
        x = self.bottleneck_conv(bottleneck)
        x = self.c1(torch.cat([self.up1(x), f3], 1))
        x = self.c2(torch.cat([self.up2(x), f2], 1))
        x = self.c3(torch.cat([self.up3(x), f1], 1))
        x = self.c4(self.up5(self.up4(x)))
        return self.out(x)

class SiameseEfficientNetUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec = EfficientNetEncoder(False), UNetDecoder()
    def forward(self, a, b):
        fa, fb = self.enc(a), self.enc(b)
        return self.dec([torch.abs(x - y) for x, y in zip(fa, fb)])

# ==================== 3. MODEL MANAGER ====================
class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_configs = {
            "Siamese Pure": {"class": SiamesePure, "raw": "models/siamese_pure_raw.pth", "processed": "models/siamese_pure_processed.pth"},
            "Siamese + MobileNetV2": {"class": SiameseUnetMobileNetV2, "raw": "models/siamese_mobilenet_raw.pth", "processed": "models/siamese_mobilenet_processed.pth"},
            "EfficientNet-B0 + U-Net": {"class": SiameseEfficientNetUNet, "raw": "models/siamese_efficientnet_unet_raw.pth", "processed": "models/siamese_efficientnet_unet_processed.pth"}
        }

    def get_model(self, model_name, data_type):
        config = self.model_configs[model_name]
        weight_path = config[data_type]
        model = config["class"]().to(self.device)
        if os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location=self.device)
            sd = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
            model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
        model.eval()
        return model

model_manager = ModelManager()

# ==================== 4. PROCESSING PIPELINES ====================

# --- ENHANCEMENT FUNCTION ---
def enhance_image(imgA, imgB):
    """
    Áp dụng CLAHE để tăng độ nét và Histogram Matching để đồng bộ màu sắc.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    imgA_lab = cv2.cvtColor(imgA, cv2.COLOR_RGB2LAB)
    imgB_lab = cv2.cvtColor(imgB, cv2.COLOR_RGB2LAB)

    imgA_lab[:, :, 0] = clahe.apply(imgA_lab[:, :, 0])
    imgB_lab[:, :, 0] = clahe.apply(imgB_lab[:, :, 0])

    imgA_clahe = cv2.cvtColor(imgA_lab, cv2.COLOR_LAB2RGB)
    imgB_clahe = cv2.cvtColor(imgB_lab, cv2.COLOR_LAB2RGB)

    try:
        imgB_matched = match_histograms(imgB_clahe, imgA_clahe, channel_axis=-1)
    except TypeError:
        imgB_matched = match_histograms(imgB_clahe, imgA_clahe, multichannel=True)
        
    return imgA_clahe.astype(np.uint8), imgB_matched.astype(np.uint8)


def get_transforms(model_name):
    if "MobileNetV2" in model_name:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([A.ToFloat(max_value=255.0), ToTensorV2()])

def run_inference_raw(model, model_name, imgA_pil, imgB_pil):
    imgA_np = np.array(imgA_pil.resize((256, 256)))
    imgB_np = np.array(imgB_pil.resize((256, 256)))
    
    # Enhancement
    imgA_np, imgB_np = enhance_image(imgA_np, imgB_np)
    
    tf = get_transforms(model_name)
    tA = tf(image=imgA_np)['image'].unsqueeze(0).to(DEVICE)
    tB = tf(image=imgB_np)['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = model(tA, tB)
        return torch.sigmoid(out)[0, 0].cpu().numpy()

def run_inference_processed(model, model_name, imgA_pil, imgB_pil):
    imgA_full = np.array(imgA_pil)
    imgB_full = np.array(imgB_pil)
    
    # Enhancement
    imgA_full, imgB_full = enhance_image(imgA_full, imgB_full)
    
    H, W, _ = imgA_full.shape
    tf = get_transforms(model_name)
    prob_map = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, H, PATCH_SIZE):
            for x in range(0, W, PATCH_SIZE):
                pA = imgA_full[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                pB = imgB_full[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                h_c, w_c = pA.shape[:2]
                
                if h_c < PATCH_SIZE or w_c < PATCH_SIZE:
                    pA = cv2.copyMakeBorder(pA, 0, PATCH_SIZE-h_c, 0, PATCH_SIZE-w_c, cv2.BORDER_CONSTANT, value=0)
                    pB = cv2.copyMakeBorder(pB, 0, PATCH_SIZE-h_c, 0, PATCH_SIZE-w_c, cv2.BORDER_CONSTANT, value=0)
                
                inA = tf(image=pA)['image'].unsqueeze(0).to(DEVICE)
                inB = tf(image=pB)['image'].unsqueeze(0).to(DEVICE)
                out = model(inA, inB)
                prob_map[y:y+h_c, x:x+w_c] = torch.sigmoid(out)[0, 0].cpu().numpy()[:h_c, :w_c]
    return prob_map

# ==================== 5. UI & GRADIO SETUP ====================

def predict_fn(imgA, imgB, model_name, data_type, threshold):
    """
    Hàm xử lý chính với ngưỡng động (Dynamic Threshold)
    """
    if imgA is None or imgB is None: return None, None, "⚠️ Vui lòng tải đủ 2 ảnh!"
    try:
        model = model_manager.get_model(model_name, data_type)
        if data_type == "raw":
            prob_map = run_inference_raw(model, model_name, imgA, imgB)
            prob_map_display = cv2.resize(prob_map, imgA.size)
        else:
            prob_map_display = run_inference_processed(model, model_name, imgA, imgB)

        # 1. DÙNG NGƯỠNG TỪ SLIDER ĐỂ TẠO MASK
        binary_mask = (prob_map_display > threshold).astype(np.uint8)
        
        heatmap = cv2.applyColorMap((prob_map_display * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        imgB_np = np.array(imgB)
        overlay = imgB_np.copy()
        if binary_mask.shape[:2] != imgB_np.shape[:2]:
            binary_mask = cv2.resize(binary_mask, (imgB_np.shape[1], imgB_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        overlay[binary_mask == 1] = [255, 0, 0]
        combined = cv2.addWeighted(imgB_np, 0.7, overlay, 0.3, 0)
        
        change_ratio = (np.sum(binary_mask) / binary_mask.size) * 100
        stats = f"**{model_name}** | Ngưỡng: **{threshold}** | Thay đổi: **{change_ratio:.2f}%**"
        return heatmap, combined, stats
    except Exception as e:
        traceback.print_exc()
        return None, None, f"❌ Lỗi: {str(e)}"

css = """
.result-image { height: 350px !important; }
.input-image { height: 300px !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("## 🛰️ Satellite Change Detection")
    
    with gr.Row():
        # CỘT TRÁI: INPUT & CẤU HÌNH
        with gr.Column(scale=4):
            with gr.Group():
                with gr.Row():
                    model_sel = gr.Dropdown(["Siamese Pure", "Siamese + MobileNetV2", "EfficientNet-B0 + U-Net"], value="Siamese + MobileNetV2", label="Mô hình", show_label=True)
                    data_sel = gr.Radio(["raw", "processed"], value="processed", label="Chế độ xử lý", show_label=True)
                
                # --- NEW SLIDER COMPONENT ---
                threshold_slider = gr.Slider(
                    minimum=0.1, 
                    maximum=0.9, 
                    value=0.5, 
                    step=0.05, 
                    label="Ngưỡng Tin Cậy (Confidence Threshold)",
                    info="Cao (>0.5): Chính xác cao (ít nhiễu) | Thấp (<0.5): Nhạy hơn (bắt chi tiết nhỏ)"
                )
                
                with gr.Row():
                    img_a = gr.Image(label="Ảnh Trước (A)", type="pil", elem_classes="input-image")
                    img_b = gr.Image(label="Ảnh Sau (B)", type="pil", elem_classes="input-image")
                
                btn_run = gr.Button("🚀 PHÂN TÍCH NGAY", variant="primary")

            # --- PHẦN EXAMPLES ---
            gr.Markdown("### 📂 Ảnh mẫu (Click để chọn)")
            gr.Examples(
                examples=[
                    # Format: [Path A, Path B, Model, Mode, Threshold]
                    ["data/examples/img_A_01.png", "data/examples/img_B_01.png", "Siamese + MobileNetV2", "processed", 0.5],
                    ["data/examples/img_A_02.png", "data/examples/img_B_02.png", "EfficientNet-B0 + U-Net", "processed", 0.6],
                    ["data/examples/img_A_03.png", "data/examples/img_B_03.png", "Siamese Pure", "raw", 0.4],
                ],
                inputs=[img_a, img_b, model_sel, data_sel, threshold_slider],
                label=None
            )

        # CỘT PHẢI: KẾT QUẢ
        with gr.Column(scale=5):
            res_stats = gr.Markdown("### ⏳ Đang chờ kết quả...")
            with gr.Row():
                res_heatmap = gr.Image(label="Heatmap (Mức độ thay đổi)", elem_classes="result-image", show_download_button=True)
                res_overlay = gr.Image(label="Overlay (Vùng thay đổi)", elem_classes="result-image", show_download_button=True)

    # Thêm slider vào danh sách input của sự kiện click
    btn_run.click(predict_fn, inputs=[img_a, img_b, model_sel, data_sel, threshold_slider], outputs=[res_heatmap, res_overlay, res_stats])

if __name__ == "__main__":
    demo.launch()