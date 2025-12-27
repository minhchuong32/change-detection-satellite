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
import matplotlib.pyplot as plt
import io

# ==================== 1. CONFIG & UTILS ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 256

def get_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==================== 2. MODEL ARCHITECTURES ====================

# --- Common Blocks ---
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, 1, 1),
            nn.BatchNorm2d(o), nn.ReLU(),
            nn.Conv2d(o, o, 3, 1, 1),
            nn.BatchNorm2d(o), nn.ReLU(),
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
        f1 = self.c1(x)
        f2 = self.c2(self.pool(f1))
        f3 = self.c3(self.pool(f2))
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

# --- B. Siamese MobileNetV2 (Updated from Test File) ---
class SiameseUnetMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Cấu hình chuẩn: weights=None như file test yêu cầu
        base = models.mobilenet_v2(weights=None) 
        self.base_layers = base.features
        
        # Decoder configurations khớp với file test
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
        x1 = self.base_layers[:2](x)    # 128x128
        x2 = self.base_layers[2:4](x1)  # 64x64
        x3 = self.base_layers[4:7](x2)  # 32x32
        x4 = self.base_layers[7:14](x3) # 16x16
        x5 = self.base_layers[14:](x4)  # 8x8
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
        self.up4, self.up5, self.c4 = nn.ConvTranspose2d(32, 16, 2, 2), nn.ConvTranspose2d(16, 16, 2, 2), DoubleConv(16, 16)
        self.out = nn.Conv2d(16, 1, 1)
    def forward(self, f):
        f1, f2, f3, bnet = f
        x = self.c1(torch.cat([self.up1(self.bottleneck_conv(bnet)), f3], 1))
        x = self.c2(torch.cat([self.up2(x), f2], 1))
        x = self.c3(torch.cat([self.up3(x), f1], 1))
        return self.out(self.c4(self.up5(self.up4(x))))

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
        self.models = {}
        self.load_configs()

    def load_configs(self):
        # Đường dẫn file weights (bạn cần đảm bảo file tồn tại)
        self.model_configs = {
            "Siamese Pure": {
                "class": SiamesePure,
                "raw": "models/siamese_pure_raw.pth",
                "processed": "models/siamese_pure_processed.pth",
            },
            "Siamese + MobileNetV2": {
                "class": SiameseUnetMobileNetV2,
                "raw": "models/siamese_mobilenet_raw.pth",
                "processed": "models/siamese_mobilenet_processed.pth",
            },
            "EfficientNet-B0 + U-Net": {
                "class": SiameseEfficientNetUNet,
                "raw": "models/siamese_efficientnet_unet_raw.pth",
                "processed": "models/siamese_efficientnet_unet_processed.pth",
            }
        }

    def get_model(self, model_name, data_type):
        config = self.model_configs[model_name]
        model_class = config["class"]
        weight_path = config[data_type]
        
        # Load Architecture
        model = model_class().to(self.device)
        
        # Load Weights
        if os.path.exists(weight_path):
            try:
                checkpoint = torch.load(weight_path, map_location=self.device)
                state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
                
                # Fix lỗi DataParallel (nếu có prefix module.)
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
                print(f"✅ Loaded: {weight_path}")
            except Exception as e:
                print(f"❌ Load Error {weight_path}: {e}")
        else:
            print(f"⚠️ Weights not found: {weight_path}, using random weights.")
            
        model.eval()
        return model

model_manager = ModelManager()

# ==================== 4. PROCESSING PIPELINES ====================

def run_inference_raw(model, imgA_pil, imgB_pil):
    """
    Pipeline cho RAW: Resize về 256x256 -> Predict
    """
    # 1. Resize cứng về 256x256
    imgA = np.array(imgA_pil.resize((256, 256)))
    imgB = np.array(imgB_pil.resize((256, 256)))
    
    # 2. Normalize cơ bản (0-1)
    tA = torch.from_numpy(imgA).float().permute(2,0,1).unsqueeze(0).to(DEVICE) / 255.0
    tB = torch.from_numpy(imgB).float().permute(2,0,1).unsqueeze(0).to(DEVICE) / 255.0
    
    # 3. Predict
    with torch.no_grad():
        out = model(tA, tB)
        prob = torch.sigmoid(out)[0, 0].cpu().numpy()
        
    return prob

def run_inference_processed(model, imgA_pil, imgB_pil):
    """
    Pipeline cho PROCESSED: Tiling 256x256 -> Padding -> Predict -> Stitching
    """
    # 1. Convert to Array (Giữ nguyên kích thước gốc)
    imgA_full = np.array(imgA_pil)
    imgB_full = np.array(imgB_pil)
    H, W, _ = imgA_full.shape
    
    # 2. Setup Transform chuẩn (ImageNet Norm)
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    prob_map = np.zeros((H, W), dtype=np.float32)
    
    # 3. Tiling Loop
    with torch.no_grad():
        for y in range(0, H, PATCH_SIZE):
            for x in range(0, W, PATCH_SIZE):
                # Cut patch
                pA = imgA_full[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                pB = imgB_full[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                # Check dimensions for Padding
                h_curr, w_curr = pA.shape[:2]
                
                # Logic Padding: Nếu patch nhỏ hơn 256x256 (ở biên), pad đen vào
                if h_curr < PATCH_SIZE or w_curr < PATCH_SIZE:
                    pad_h = PATCH_SIZE - h_curr
                    pad_w = PATCH_SIZE - w_curr
                    pA = cv2.copyMakeBorder(pA, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                    pB = cv2.copyMakeBorder(pB, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                
                # Normalize & Tensor
                inA = transform(image=pA)['image'].unsqueeze(0).to(DEVICE)
                inB = transform(image=pB)['image'].unsqueeze(0).to(DEVICE)
                
                # Predict
                out = model(inA, inB)
                prob_patch = torch.sigmoid(out)[0, 0].cpu().numpy()
                
                # Cắt bỏ phần padding (nếu có) để ghép lại map gốc
                valid_prob = prob_patch[:h_curr, :w_curr]
                prob_map[y:y+h_curr, x:x+w_curr] = valid_prob
                
    return prob_map

# ==================== 5. MAIN GRADIO FUNCTION ====================

def predict_fn(imgA, imgB, model_name, data_type):
    if imgA is None or imgB is None:
        return None, None, "⚠️ Vui lòng tải đủ 2 ảnh!"
    
    try:
        # Load Model
        model = model_manager.get_model(model_name, data_type)
        
        # Chọn chiến lược xử lý dựa trên data_type
        if data_type == "raw":
            # Chiến lược 1: Resize 256x256
            prob_map = run_inference_raw(model, imgA, imgB)
            # Resize mask lại về kích thước gốc để hiển thị overlay đẹp hơn (optional)
            prob_map_display = cv2.resize(prob_map, imgA.size) 
        else:
            # Chiến lược 2: Tiling + Padding + Norm
            prob_map = run_inference_processed(model, imgA, imgB)
            prob_map_display = prob_map

        # --- VISUALIZATION ---
        # 1. Binary Mask (Fixed Threshold 0.5 - Yêu cầu bỏ slider đầu vào)
        binary_mask = (prob_map_display > 0.5).astype(np.uint8)
        
        # 2. Heatmap (Jet)
        heatmap = cv2.applyColorMap((prob_map_display * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 3. Overlay
        imgB_np = np.array(imgB)
        # Nếu là RAW, imgB_np có thể to hơn mask (do mask resize 256), cần resize imgB về mask hoặc ngược lại
        # Ở đây ta resize mask về kích thước ảnh gốc imgB để overlay chính xác
        if prob_map_display.shape[:2] != imgB_np.shape[:2]:
             prob_map_display = cv2.resize(prob_map_display, (imgB_np.shape[1], imgB_np.shape[0]))
             binary_mask = (prob_map_display > 0.5).astype(np.uint8)

        overlay = imgB_np.copy()
        overlay[binary_mask == 1] = [255, 0, 0] # Tô đỏ vùng thay đổi
        combined = cv2.addWeighted(imgB_np, 0.7, overlay, 0.3, 0)
        
        # Stats
        change_ratio = (np.sum(binary_mask) / binary_mask.size) * 100
        stats = f"""
        ### ✅ Kết quả Phân tích
        - **Model:** {model_name}
        - **Chế độ:** {data_type.upper()}
        - **Kích thước ảnh gốc:** {imgB.size}
        - **Tỷ lệ thay đổi:** {change_ratio:.2f}%
        - **Ngưỡng cố định:** 0.5
        """
        
        return heatmap, combined, stats

    except Exception as e:
        traceback.print_exc()
        return None, None, f"❌ Lỗi: {str(e)}"

# ==================== 6. UI SETUP ====================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛰️ Satellite Change Detection System")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Cấu hình")
            model_sel = gr.Radio(
                ["Siamese Pure", "Siamese + MobileNetV2", "EfficientNet-B0 + U-Net"],
                value="Siamese + MobileNetV2", 
                label="Chọn Mô hình"
            )
            data_sel = gr.Radio(
                ["raw", "processed"], 
                value="processed", 
                label="Loại Dữ liệu (Chiến lược xử lý)"
            )
            gr.Info("Raw: Resize 256x256 | Processed: Tiling + Padding + Norm")
            
            gr.Markdown("### 2. Input Images")
            img_a = gr.Image(label="Ảnh Trước (Before)", type="pil", height=300)
            img_b = gr.Image(label="Ảnh Sau (After)", type="pil", height=300)
            
            btn_run = gr.Button("🚀 Phân tích Thay đổi", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 3. Kết quả")
            res_heatmap = gr.Image(label="Heatmap (Xác suất)", type="numpy")
            res_overlay = gr.Image(label="Overlay (Vùng thay đổi)", type="numpy")
            res_stats = gr.Markdown("Waiting for input...")

    btn_run.click(
        predict_fn, 
        inputs=[img_a, img_b, model_sel, data_sel],
        outputs=[res_heatmap, res_overlay, res_stats]
    )

if __name__ == "__main__":
    demo.launch()