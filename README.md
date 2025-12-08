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
| Foggy  | 0.212 | 0.213 | 0.957 |

---

## Repository Structure
CARLA_PERCEPTION_DATASET/<br>
├── carla_dataset/<br>
│   ├── clear/<br>
│   │   └── calib/                 # Camera calibration files (intrinsics & extrinsics)<br>
│   ├── rainy/<br>
│   │   └── calib/<br>
│   └── foggy/<br>
│       └── calib/<br>
│<br>
├── multi_images/<br>
│   ├── clear/<br>
│   │   └── scene_0001/             # Multi-view RGB images for clear weather<br>
│   ├── rainy/<br>
│   │   └── scene_0001/             # Multi-view RGB images for rainy weather<br>
│   └── foggy/<br>
│       └── scene_0001/             # Multi-view RGB images for foggy weather<br>
│<br>
├── pseudo_labels/<br>
│   ├── clear/<br>
│   │   └── scene_0001/             # BEV pseudo-labels for clear weather<br>
│   ├── rainy/<br>
│   │   └── scene_0001/             # BEV pseudo-labels for rainy weather<br>
│   └── foggy/<br>
│       └── scene_0001/             # BEV pseudo-labels for foggy weather<br>
│<br>
BEVUNet.ipynb                   # Main notebook: model, training, evaluation<br>
CARLA.ipynb                     # CARLA data preprocessing and handling<br>
bev_unet_multiview_best.pth     # Trained BEVUNet model checkpoint<br>
README.md                       # Project documentation<br>


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
model.load_state_dict(torch.load("bev_unet_multiview_best.pth"))
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

This work was developed as part of research on robust camera-only BEVUNet perception under adverse weather conditions. <br>

**Team members:**
1. Pranav Jain<br>
2. Avantika Rattan<br>
3. Venkatakrishnan Amruthur Kesavan<br>

