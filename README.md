# BEVUNet: Camera-Only BEV Occupancy and Detection Under Adverse Weather

This repository contains the implementation of **BEVUNet**, a lightweight camera-only Bird’s-Eye-View (BEV) perception network designed for **occupancy prediction and object detection** under **clear, rainy, and foggy weather conditions**. The model follows a U-Net–style encoder–decoder architecture with skip connections and is trained using **weak supervision with pseudo-labels**.

---

## Key Features
- Camera-only BEV perception (no LiDAR required)
- Lightweight U-Net–style encoder–decoder
- Supports clear, rainy, and foggy conditions
- BEV occupancy segmentation and object detection
- Weather-adaptive inference thresholds
- Weakly supervised learning using pseudo-labels

---

## Network Architecture
- **Input:** 18-channel stacked multi-view features
- **Encoder:** 3 stages (32 → 64 → 128 channels) with max pooling
- **Decoder:** 2 stages with transposed convolution upsampling and skip connections
- **Output:** BEV occupancy logits (1 × 128 × 128)

The architecture is intentionally lightweight while remaining competitive with heavier transformer-based BEV models.

---

## Performance Summary

### Detection Performance (mAP)
| Weather | mAP |
|-------|-----|
| Clear | 0.768 |
| Rainy | 0.774 |
| Foggy | 0.730 |
| Overall | 0.757 |

### BEV Occupancy Segmentation
| Weather | mIoU | Precision | Recall |
|--------|------|-----------|--------|
| Clear  | 0.419 | 0.437 | 0.936 |
| Rainy  | 0.251 | 0.253 | 0.971 |
| Foggy  | 0.212 | 0.213 | 0.958 |

---

## Repository Structure
CARLA_PERCEPTION_DATASET/
├── carla_dataset/
│   ├── clear/
│   │   └── calib/                 # Camera calibration files (intrinsics & extrinsics)
│   ├── rainy/
│   │   └── calib/
│   └── foggy/
│       └── calib/
│
├── multi_images/
│   ├── clear/
│   │   └── scene_0001/             # Multi-view RGB images for clear weather
│   ├── rainy/
│   │   └── scene_0001/             # Multi-view RGB images for rainy weather
│   └── foggy/
│       └── scene_0001/             # Multi-view RGB images for foggy weather
│
├── pseudo_labels/
│   ├── clear/
│   │   └── scene_0001/             # BEV pseudo-labels for clear weather
│   ├── rainy/
│   │   └── scene_0001/             # BEV pseudo-labels for rainy weather
│   └── foggy/
│       └── scene_0001/             # BEV pseudo-labels for foggy weather
│
├── BEVUNet.ipynb                   # Main notebook: model, training, evaluation
├── CARLA.ipynb                     # CARLA data preprocessing and handling
├── bev_unet_multiview_best.pth     # Trained BEVUNet model checkpoint
└── README.md                       # Project documentation


---

## Installation

### Create Anaconda Environment
```bash
conda create -n bevunet python=3.9 -y
conda activate bevunet
```

---

## Saving and Loading the Model

### Save

```python
torch.save(model.state_dict(), "bev_unet_multiview_best.pth")
```

### Load

```python
model = BEVUNet(in_ch=18, base_ch=32)
model.load_state_dict(torch.load("bevunet_weights.pth"))
model.eval()
```

---

## Method Overview

BEVUNet learns BEV representations directly from multi-view camera features using weak supervision. Skip connections preserve fine-grained spatial details, while weather-adaptive thresholds improve robustness under adverse visibility.

---
## Limitations

1. Performance degrades under dense fog due to depth ambiguity

2. No temporal fusion across frames

3. Camera-only constraints limit geometric accuracy

---
## Future Work

1. Temporal feature aggregation

2. Uncertainty-aware occupancy prediction

3. Depth-supervised pretraining

4. Domain adaptation for severe weather

---
## License

This project is intended for academic and research use.

---
## Acknowledgements

This work was developed as part of research on robust camera-only BEV perception under adverse weather conditions.

