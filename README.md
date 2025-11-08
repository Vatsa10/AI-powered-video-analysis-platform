# AI-Powered Video Analysis: Violence Detection

This project implements a **Deep Learning-based violence detection system** using MobileNet + BiLSTM architecture, achieving **96% accuracy** on the Real-Life Violence Situations Dataset. The model was trained using Kaggle's cloud infrastructure for better performance and scalability.

## Model Details
- **Architecture**: MobileNet (feature extraction) + BiLSTM (temporal modeling)
- **Accuracy**: 96% on test set
- **Dataset**: [Real-Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
- **Input**: 16-frame video clips (224x224x3 per frame)
- **Output**: Binary classification (Violence/Non-Violence)
- **Framework**: TensorFlow/Keras
- **Training**: Performed on Kaggle Notebooks with GPU acceleration

## Project Structure
```
AI-Powered-Video-Analysis/
│
├── training_scripts/
│   └── MobileNet + Bi-LSTM.ipynb   # Training notebook for Kaggle
├── violence_model_tf/              # Saved model directory (generated after training)
├── test-model-cli.py               # Command-line interface for testing
├── app.py                          # Flask web interface
└── requirements.txt                # Project dependencies
```

## Model Training on Kaggle

1. **Upload to Kaggle**:
   - Upload the `MobileNet + Bi-LSTM.ipynb` notebook to Kaggle
   - Enable GPU acceleration in the notebook settings
   - Ensure the Real-Life Violence Dataset is added to the notebook

2. **Run the Training**:
   - Open the notebook in Kaggle
   - Run cells sequentially to:
     1. Install required dependencies
     2. Load and preprocess the dataset
     3. Define and compile the model
     4. Train the model (this may take several hours)
     5. Save the trained model

3. **Download the Model**:
   - After training completes, the model will be saved in the `violence_model_tf` directory
   - Download this directory and place it in your project root

## Usage with Pre-trained Model

### CLI Testing
```bash
python test-model-cli.py --video_path path/to/video.mp4
```

### Web Interface
```bash
python app.py
# Open http://localhost:5000 in your browser
```

## Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Vatsa10/AI-powered-video-analysis-platform
   cd AI-powered-video-analysis-platform
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies
- Python 3.7+
- TensorFlow 2.12.0+
- OpenCV 4.7.0+
- NumPy 1.21.0+
- Flask 2.0.0+ (for web interface)
- Jupyter (for running the training notebook)
- Kaggle environment (for model training)

## Notes
- The model requires the `violence_model_tf` directory containing the trained model files
- Ensure you have sufficient disk space for the dataset and model files
- For best performance, use a GPU-enabled environment for inference