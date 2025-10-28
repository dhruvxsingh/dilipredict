# DILITracer Setup and Usage Guide
## System Overview
DILITracer is an AI model for Drug-Induced Liver Injury (DILI) level prediction using liver organoid brightfield images. The model classifies compounds into three categories:

- **No-DILI**: No drug-induced liver injury concern
- **Less-DILI**: Less DILI concern
- **Most-DILI**: Most DILI concern

## ✅ Current Status

- Model architecture implemented (STViT with BEiT-V2 encoder)
- Inference pipeline ready
- Successfully tested with sample organoid images
- Ready to accept new input images for classification

## 🚀 Quick Start
### 1. Clone the Repository
```bash
git clone https://github.com/dhruvxsingh/dilipredict.git
cd dilipredict
```
### 2. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install numpy torch torchvision einops pillow scikit-learn
```
### 3. Test the System
```bash
python demo.py
```

## 📊 Running Inference on Your Images

### Image Requirements

- **Format**: TIFF images
- **Organization**: 4 time points (Day 0 to Day 3)
- **Structure**: Multiple z-stack images per day
- **Resolution**: Images will be automatically resized to 224x224

### Directory Structure
```
data/
└── your_experiment/
    ├── D00/  # Day 0 images
    │   ├── image_z001.tiff
    │   ├── image_z002.tiff
    │   └── ...
    ├── D01/  # Day 1 images
    ├── D02/  # Day 2 images
    └── D03/  # Day 3 images
```
### Running Inference
```python
from dilipredict.models import DILIPredict
from dilipredict.image_loader import ImageLoader
from dilipredict.pipelines import DILIPredict as DILIPipeline
import torch

# Initialize components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DILIPredict()
img_loader = ImageLoader()
pipeline = DILIPipeline(img_loader, model, device)

# Prepare your image paths
img_files = [
    ['data/your_experiment/D00/img1.tiff', 'data/your_experiment/D00/img2.tiff'],
    ['data/your_experiment/D01/img1.tiff', 'data/your_experiment/D01/img2.tiff'],
    ['data/your_experiment/D02/img1.tiff', 'data/your_experiment/D02/img2.tiff'],
    ['data/your_experiment/D03/img1.tiff', 'data/your_experiment/D03/img2.tiff'],
]

# Run inference
label, probabilities = pipeline(img_files)

# Interpret results
label_map = {0: "No-DILI", 1: "Less-DILI", 2: "Most-DILI"}
print(f"Predicted: {label_map[label]}")
print(f"Probabilities: No-DILI={probabilities[0]:.4f}, Less-DILI={probabilities[1]:.4f}, Most-DILI={probabilities[2]:.4f}")
```
## 🔬 Model Architecture

- Image Encoder: BEiT-V2 Vision Transformer
- Spatial Encoder: 2-layer ViT for 3D spatial relationships
- Temporal Encoder: Bidirectional LSTM for temporal dynamics
- Classification: MLP for 3-class output

## 📈 Performance

- Total Parameters: ~130M
- Input: 4 days × multiple z-stacks × 224×224 RGB images
- Output: 3-class probability distribution

## 📝 Notes

- The model currently runs with random weights (not pretrained on DILI data)
- For production use, the model should be trained on labeled DILI organoid data
- GPU recommended for faster inference