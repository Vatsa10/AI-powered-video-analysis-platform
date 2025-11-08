# AI/ML-Powered Video Analysis & Interpretation

This project demonstrates a **Deep Learning-based video analysis system** capable of:
- Detecting **violence** in real-life video footage.
- Detecting **anomalies** or abnormal motion patterns.

The project uses **multiple deep learning models**:
1. **MobileNet + BiLSTM** → High-accuracy violence detection (96% accuracy)  
2. **X3D (Action Recognition)** → Detects violence or aggressive behavior.  
3. **ConvLSTM Autoencoder** → Detects anomalies based on frame reconstruction error.

---

## Features
- Automated **video frame extraction** and dataset handling.
- Dual deep learning model setup for violence & anomaly detection.
- Modular & easy to extend for additional video analytics.
- Ready for real-time deployment using live camera feeds.

---

## Model Overview

| Model | Task | Dataset | Input Shape | Framework |
|-------|------|----------|--------------|------------|
| **MobileNet + BiLSTM** | Violence Detection | [Real-Life Violence Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) | (16, 224, 224, 3) | TensorFlow/Keras |
| X3D-xs | Violence / Action Recognition | Real-Life Violence Dataset (Kaggle) | 3x16x112x112 | PyTorch |
| ConvLSTM Autoencoder | Anomaly Detection | UCSD Pedestrian Dataset | 10x3x64x64 | PyTorch |

---

## Folder Structure

```
AI-Powered-Video-Analysis/
│
├── datasets/
│   ├── violence_dataset/               # Real-life Violence Dataset (Kaggle)
│   ├── ucsd_anomaly_dataset/           # UCSD Anomaly Detection Dataset
│   ├── processed/
│
├── scripts/
│   ├── prepare_dataset.py              # Frame extraction & dataset preparation
│   ├── setup_env.bat                   # Windows environment setup script
│   ├── setup_env.sh                    # Linux/Mac environment setup script
│   ├── setup_env.py                    # Cross-platform environment setup script
│
├── training scripts/
│   └── MobileNet + Bi-LSTM.ipynb       # Training notebook for MobileNet+BiLSTM model
├── test-model-cli.py                   # CLI script for model testing
├── app.py                              # Flask API and web interface
│
├── models/
│   ├── x3d_model.py                    # X3D-based Action Recognition Model
│   ├── convlstm_autoencoder.py         # ConvLSTM Autoencoder for anomaly detection
│
├── train/
│   ├── train_x3d.py                    # Training pipeline for violence detection
│   ├── train_convlstm.py               # Training pipeline for anomaly detection
│
├── inference_pipeline.py               # Unified analysis on test videos
├── requirements.txt
└── README.md
```

---

## Datasets

### Real-Life Violence Situations Dataset
[Kaggle Dataset Link](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)

**Structure:**
```
violence_dataset/
├── Violence/
├── NonViolence/
```

### UCSD Anomaly Detection Dataset
[UCSD Dataset Link](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)

**Structure:**
```
ucsd_anomaly_dataset/
├── Train/
├── Test/
```

---

## Module 1: MobileNet + BiLSTM for Violence Detection

### Model Details
- **Architecture**: MobileNet (feature extraction) + BiLSTM (temporal modeling)
- **Accuracy**: 96% on test set
- **Input**: 16-frame video clips (224x224x3 per frame)
- **Output**: Binary classification (Violence/Non-Violence)

### Usage

#### Training
```bash
# Run the training notebook
jupyter notebook "training scripts/MobileNet + Bi-LSTM.ipynb"
```

#### CLI Testing
```bash
python test-model-cli.py --video_path path/to/video.mp4
```

#### Web Interface
```bash
python app.py
# Open http://localhost:5000 in your browser
```

## Setup Instructions

### Clone the repository
```bash
git clone https://github.com/Vatsa10/AI-powered-video-analysis-platform
cd AI-powered-video-analysis-platform
```

### Automated Environment Setup
The project includes scripts to automatically set up a virtual environment, install uv, and install all dependencies:

**Windows:**
```bash
scripts\setup_env.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

**Cross-platform (Python):**
```bash
python scripts/setup_env.py
```

### Manual Setup
Alternatively, you can manually set up the environment:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install uv:
   ```bash
   pip install uv
   ```

4. Install project dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

### Prepare dataset (convert videos → frames)
```bash
python scripts/prepare_dataset.py
```

This extracts frames from videos and organizes them into class-wise folders for training.

---

## Model Training

### Train Violence Detection (X3D)
```bash
python train/train_x3d.py
```

This trains an X3D model for binary classification: *Violence* vs *Non-Violence*.

### Train Anomaly Detection (ConvLSTM AE)
```bash
python train/train_convlstm.py
```

This trains an Autoencoder to learn normal motion. High reconstruction error = anomaly.

---

## Inference

### Run unified inference pipeline
```bash
python inference_pipeline.py --video test_video.mp4
```

**Outputs:**
* `"Violence Detected"` or `"No Violence"`
* `Anomaly Score` (higher = abnormal activity)

---

## Requirements

```
torch
torchvision
opencv-python
tqdm
numpy
scikit-learn
pytorch-lightning
pillow
matplotlib
pandas
```

---

## Future Improvements

* Integrate **YOLOv8** or **MViT** for object & action recognition.
* Add **real-time camera inference** using OpenCV's VideoCapture.
* Deploy as a **Flask / FastAPI web service** with live video feeds.

---

## Optional Upgrade Path

Once your prototype runs successfully:
* Integrate **live webcam streaming** inference (`cv2.VideoCapture(0)`).
* Use **Grad-CAM** for explainability of detected violence frames.
* Deploy model inference as an **API endpoint** with **FastAPI** or **Flask**.