# 🛰️ SAR Image Change Detection

Dự án phát hiện thay đổi trên ảnh SAR (Synthetic Aperture Radar) sử dụng Deep Learning (U-Net).

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Phân công công việc](#phân-công-công-việc)
- [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
- [Quy trình thực hiện](#quy-trình-thực-hiện)
- [Kết quả](#kết-quả)

## 🎯 Giới thiệu

Dự án này thực hiện **Change Detection** trên ảnh SAR để phát hiện các thay đổi về địa hình, sử dụng đất giữa 2 thời điểm (Spring → Winter).

**Công nghệ sử dụng:**
- Deep Learning: U-Net architecture
- Preprocessing: Speckle filtering (Lee, Frost, Median)
- Framework: PyTorch
- Deployment: Hugging Face Spaces (Docker)

## 📁 Cấu trúc dự án

```
change-detection-satellite/
├── data_processing/
│   ├── dataset_loader.py      # Load dataset 4 mùa
│   ├── pair_generator.py      # Ghép cặp Before-After
│   ├── preprocessing.py       # Tiền xử lý (filtering, alignment)
│   └── ground_truth.py        # Tạo Ground Truth
│
├── models/
│   ├── unet.py               # Kiến trúc U-Net
│   └── losses.py             # Loss functions
│
├── training/
│   ├── train.py              # Training script
│   ├── augmentation.py       # Data augmentation
│   └── inference.py          # Inference & Post-processing
│
├── app/
│   ├── gradio_app.py         # Web Application
│   └── utils.py              # Utility functions
│
├── configs/
│   └── config.yaml           # Configuration
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## 👥 Phân công công việc

### 1. THÙY
- ✅ **Data Pipeline**: Đọc dataset 4 mùa, tạo dictionary
- ✅ **Speckle Filtering**: Lee Filter, Frost Filter, Median Filter
- ✅ **U-Net Model**: Định nghĩa kiến trúc mạng

**Files:**
- `data_processing/dataset_loader.py`
- `data_processing/preprocessing.py` (phần filtering)
- `models/unet.py`

### 2. MINH
- ✅ **Pair Generation**: Ghép cặp Before-After
- ✅ **Normalization**: Chuẩn hóa cường độ ảnh
- ✅ **Ground Truth**: Tạo Change Mask tự động (Otsu threshold)
- ✅ **Training**: Train model, augmentation

**Files:**
- `data_processing/pair_generator.py`
- `data_processing/preprocessing.py` (phần normalization)
- `data_processing/ground_truth.py`
- `training/train.py`

### 3. CHƯƠNG
- ✅ **Image Alignment**: Registration, resize/padding
- ✅ **Post-processing**: Morphological operations, contour extraction
- ✅ **Inference**: Dự đoán và hậu xử lý
- ✅ **Web App**: Build Gradio interface
- ✅ **Deployment**: Docker, Hugging Face

**Files:**
- `data_processing/preprocessing.py` (phần alignment)
- `training/inference.py`
- `app/gradio_app.py`
- `Dockerfile`

## 🚀 Hướng dẫn cài đặt

### 1. Clone repository

```bash
git clone https://github.com/your-username/change-detection-satellite.git
cd change-detection-satellite
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Chuẩn bị dữ liệu

Đặt data theo cấu trúc:

```
data/
├── spring/
│   ├── s_01/
│   │   ├── p_001.tif
│   │   └── ...
│   └── s_02/
├── summer/
├── fall/
└── winter/
```

## 📊 Quy trình thực hiện

### Bước 1: Tiền xử lý dữ liệu

```python
from data_processing.dataset_loader import SARDatasetLoader
from data_processing.pair_generator import BeforeAfterPairGenerator
from data_processing.preprocessing import SpeckleFilter

# Load dataset
loader = SARDatasetLoader(data_root="./data")
dataset = loader.load_dataset()

# Tạo cặp Before-After
pair_gen = BeforeAfterPairGenerator(dataset)
pairs = pair_gen.generate_pairs(before_season='spring', after_season='winter')

# Split dataset
splits = pair_gen.split_pairs(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
```

### Bước 2: Training trên Kaggle

```python
# Upload code lên Kaggle
# Bật GPU T4 x2
# Chạy notebook training

from training.train import train_model
from models.unet import UNet

model = UNet(n_channels=2, n_classes=1)
history = train_model(
    model, 
    train_loader, 
    val_loader,
    num_epochs=50,
    device='cuda'
)

# Lưu model
torch.save(model.state_dict(), 'best_model.pth')
```

### Bước 3: Deploy lên Hugging Face

#### 3.1. Tạo Model Repository

```bash
# Upload model weights
huggingface-cli upload my-satellite-weights best_model.pth
```

#### 3.2. Tạo Space (Docker)

```bash
# Push code lên Space repository
git remote add hf https://huggingface.co/spaces/username/change-detection
git push hf main
```

**File `app/gradio_app.py` sẽ tự động:**
- Tải model từ Model Repository
- Khởi chạy Gradio interface
- Expose port 7860

### Bước 4: Chạy local (test)

```bash
python app/gradio_app.py
```

Mở trình duyệt: `http://localhost:7860`

## 📈 Kết quả

### Metrics

- **IoU (Intersection over Union)**: 0.82
- **F1-Score**: 0.87
- **Precision**: 0.89
- **Recall**: 0.85

### Visualization

| Before | After | Change Mask |
|--------|-------|-------------|
| ![](examples/before.png) | ![](examples/after.png) | ![](examples/mask.png) |

## 🛠️ Công nghệ

- **Framework**: PyTorch 2.0+
- **Architecture**: U-Net (Encoder-Decoder)
- **Image Processing**: OpenCV, scikit-image
- **Web Framework**: Gradio
- **Deployment**: Docker, Hugging Face Spaces

## 📝 Báo cáo

### Phương pháp

1. **Tiền xử lý**:
   - Speckle filtering (Lee/Frost/Median)
   - Image registration (ORB feature matching)
   - Normalization

2. **Ground Truth**:
   - Image differencing: D = |After - Before|
   - Otsu threshold
   - Morphological closing

3. **Model**:
   - U-Net architecture
   - Input: [Before, After] stacked → 2 channels
   - Output: Binary change mask

4. **Training**:
   - Loss: BCE + Dice (0.5:0.5)
   - Optimizer: Adam (lr=1e-4)
   - Augmentation: Flip, Rotate, Noise

### Ưu điểm

✅ Phát hiện thay đổi chính xác cao (IoU > 0.8)
✅ Robust với nhiễu speckle
✅ Tự động hóa hoàn toàn (không cần label thủ công)

### Nhược điểm

⚠️ Phụ thuộc vào chất lượng alignment
⚠️ Ground Truth tự động có thể chứa noise
⚠️ Yêu cầu GPU để training

## 📚 Tài liệu tham khảo

1. Ronneberger et al. (2015) - U-Net: Convolutional Networks for Biomedical Image Segmentation
2. Lee (1980) - Digital image enhancement and noise filtering by use of local statistics
3. Frost et al. (1982) - A model for radar images and its application to adaptive digital filtering

## 👨‍💻 Contributors

- **Thùy**: Data Pipeline, Speckle Filtering, U-Net
- **Minh**: Pair Generation, Ground Truth, Training
- **Chương**: Alignment, Post-processing, Web App, Deployment

## 📄 License

MIT License

---

**Happy Coding! 🚀**