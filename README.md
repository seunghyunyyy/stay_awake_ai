# Stay Awake AI 😴➡️🟢

A pipeline for **camera-based drowsiness detection**: model training, **TFLite** export, and **real-time inference** (video/webcam), plus lightweight log analysis.

> This repo targets the stack in `requirements.txt` (e.g., TensorFlow 2.20, OpenCV 4.12, PyTorch 2.8 CUDA 12.x). Adjust as needed for your machine (CPU-only vs GPU).

---

## ✨ Features

- **Training**: Train a drowsiness classifier on facial cues.
- **Inference**:
  - Real-time webcam or video file with TFLite  
  - Thresholding, label mapping, smoothing options
- **Analysis**: Read CSV prediction logs and plot quick stats.

---

## 📚 Dataset

This project uses the Kaggle **Drowsy Detection Dataset** by *Yashar Jebraeily*:

- Dataset: https://www.kaggle.com/datasets/yasharjebraeily/drowsy-detection-dataset

> Make sure to follow the dataset’s license/usage terms. If you publish results, please cite the dataset accordingly.

---

## 📦 Environment

### 1) Create a virtual environment
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Notes**
- Key packages (from `requirements.txt`):
  - TensorFlow 2.20.x (+ `tf_keras`), OpenCV 4.12, Pandas/Matplotlib
  - PyTorch 2.8.0 + CUDA 12.9 build, torchvision 0.23.0 + CUDA 12.9
  - NVIDIA CUDA 12.x user-space libs (for GPU setups)
- **CPU-only** users: install CPU wheels (e.g., `torch==2.8.0+cpu`) or remove CUDA-specific entries.

---

## 🗂️ Project Layout (example)

```
stay_awake_ai/
├─ dataset/                 # Your dataset root (train/val/test or similar)
├─ export/                  # Training outputs (checkpoints, logs)
├─ export_tflite/           # TFLite artifacts (.tflite, labels.json)
├─ train.py                 # Training script
├─ export_tflite.py         # TFLite conversion script
├─ tflite_test.py           # TFLite inference (video/webcam)
├─ analyze.py               # CSV log analysis & plots
└─ requirements.txt
```

---

## 🚀 Quickstart

### A) Real-time / video inference with **TFLite**
```bash
# video file
python tflite_test.py export_to_tflite/drowsy_fp16.tflite export/labels.json input.mp4 --show

# webcam (index 0) + mirrored preview
python tflite_test.py export_to_tflite/drowsy_fp16.tflite export/labels.json --camera 0 --show --flip
```
Common flags:
- `--decision-thr 0.8` → classify as DROWSY if p≥0.8  
- `--drowsy-label DROWSY` → label name for positive class  
- `--no-csv` → disable CSV logging

### B) Analyze predictions
```bash
python analyze.py ./result/Non_Drowsy.csv --plot
```

### C) Train a model
```bash
python train.py \
  --data-root ./dataset \
  --epochs 20 \
  --batch-size 32 \
  --img-size 224 \
  --out ./export
```

### D) Export to **TFLite**
```bash
python export_tflite.py \
  --ckpt ./export/best_model.h5 \
  --out ./export_tflite/drowsy_fp16.tflite \
  --quantize fp16
```

> Check each script’s `--help` for the exact argument set implemented in your repo.

---

## 🔧 Troubleshooting

- **Webcam not found**: try `--camera 0/1/2…`
- **Low FPS**: lower input resolution, use FP16/INT8 quantization, or frame-skip
- **Label / preprocessing mismatch**: ensure train & inference use identical resize/normalize rules
- **CUDA mismatch**: match driver/toolkit and wheels; on servers without proper CUDA, switch to CPU wheels

---

## 🗺️ Roadmap (example)

- [ ] **Late Fusion**: combine camera predictions with smartwatch heart-rate (post-hoc fusion)
- [ ] **Log pairing**: auto-pair face/HR logs at the same timestamps → build a paired dataset
- [ ] **Early/Hybrid Fusion**: train a multimodal model once paired data is collected

---

## 🤝 Contributing

Issues and PRs are welcome. Please include:
- exact commands you ran
- environment info (OS, Python, GPU/CPU)
- minimal repro data paths (or synthetic samples)
