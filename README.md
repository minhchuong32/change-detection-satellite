---
title: SAR Change Detection Multi-Model System
emoji: 🛰️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🛰️ SAR Change Detection - Multi-Model Comparison System

Hệ thống so sánh 3 kiến trúc Deep Learning cho bài toán phát hiện thay đổi trên ảnh vệ tinh SAR.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YOUR_USERNAME/sar-change-detection)

## 🎯 Tính năng

✅ **3 Model Architectures:**
- Siamese Pure (CNN thuần túy)
- Siamese + MobileNetV2 (Transfer Learning)
- EfficientNet-B0 + U-Net (State-of-the-art)

✅ **So sánh Raw vs Processed Data:**
- Raw: LEVIR-CD+ original
- Processed: Đã qua speckle filtering, alignment, normalization

✅ **Real-time Inference:**
- Upload 2 ảnh SAR (before/after)
- Điều chỉnh confidence threshold
- Xem heatmap + overlay + metrics

✅ **Comprehensive Metrics:**
- IoU, F1-Score, Precision, Recall
- Biểu đồ so sánh Raw vs Processed
- Bảng so sánh 6 variants

---

## 📊 Kết quả

| Model | Data Type | IoU | F1 | Precision | Recall |
|-------|-----------|-----|----|-----------| -------|
| **EfficientNet U-Net** | Processed | **0.856** | **0.908** | **0.916** | **0.901** |
| EfficientNet U-Net | Raw | 0.781 | 0.849 | 0.861 | 0.837 |
| MobileNetV2 | Processed | 0.834 | 0.892 | 0.901 | 0.883 |
| MobileNetV2 | Raw | 0.756 | 0.831 | 0.842 | 0.820 |
| Siamese Pure | Processed | 0.812 | 0.876 | 0.889 | 0.864 |
| Siamese Pure | Raw | 0.723 | 0.802 | 0.815 | 0.789 |

**🏆 Best Model:** EfficientNet-B0 + U-Net (Processed) - F1 Score: 0.908

---

## 🗂️ Datasets

### Raw Dataset
- **Source:** [LEVIR-CD+](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd-change-detection)
- **Size:** 10,192 image pairs (1024×1024)
- **Coverage:** Texas, USA (2002-2020)
- **Change types:** Construction, demolition, expansion

### Processed Dataset
- **Source:** [Satellite Change Detection](https://www.kaggle.com/datasets/nguynthanhbnhminh/satellite-dataset-for-change-detection)
- **Preprocessing:**
  - Speckle filtering (Lee, Frost, Median)
  - Image alignment (ORB + RANSAC)
  - Normalization & contrast enhancement
  - Data augmentation

---

## 🛠️ Technologies

- **Framework:** PyTorch 2.1.0
- **Pre-trained Models:** timm (EfficientNet, MobileNetV2)
- **Image Processing:** OpenCV, PIL
- **Web Interface:** Gradio 4.44.0
- **Deployment:** Hugging Face Spaces

---

## 👥 Team

**Đồ án cuối kì Xử Lý Ảnh Số - Nhóm 03**

- **🔵 Thùy:** Data Pipeline, Speckle Filtering, Siamese Pure
- **🟢 BiMi:** Pair Generation, Ground Truth, MobileNetV2 Integration
- **🟣 Chương:** Image Alignment, Post-processing, EfficientNet U-Net, Deployment

---

## 📚 References

1. **LEVIR-CD:** Chen et al. (2020) - Remote Sensing Change Detection
2. **U-Net:** Ronneberger et al. (2015) - Biomedical Image Segmentation
3. **EfficientNet:** Tan & Le (2019) - Compound Model Scaling
4. **MobileNetV2:** Sandler et al. (2018) - Inverted Residuals

---

## 📄 License

MIT License - Free for academic and commercial use

---

## 🔗 Links

- **Kaggle Training Notebooks:** [View Code](https://kaggle.com/)
- **GitHub Repository:** [Source Code](https://github.com/)
- **Paper (Coming Soon):** Change Detection on SAR Images

---
## Cau truc
sar-change-detection/
│
├── app.py                          # Main application (code bạn đã có)
├── requirements.txt                # Fixed dependencies
├── README.md                       # Hugging Face README
├── .gitattributes                  # Git LFS config
│
├── models/                         # Model weights (sẽ dùng Git LFS)
│   ├── siamese_pure_raw.pth
│   ├── siamese_pure_processed.pth
│   ├── siamese_mobilenet_raw.pth
│   ├── siamese_mobilenet_processed.pth
│   ├── siamese_efficientnet_unet_raw.pth
│   └── siamese_efficientnet_unet_processed.pth
│
├── metrics.json                    # Metrics data
│
└── data/                           # (Optional) Example images
    └── examples/
        ├── img_A_01.png
        ├── img_B_01.png
        ├── img_A_02.png
        ├── img_B_02.png
        ├── img_A_03.png
        └── img_B_03.png
**Made with ❤️ by Team Thùy-BiMi-Chương**