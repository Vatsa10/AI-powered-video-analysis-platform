import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# ✅ Load trained model
model_path = "my_model.keras"
model = load_model(model_path)
print("✅ Model loaded successfully!")
print("Model expects input shape:", model.input_shape)

# Dynamically extract input dimensions
_, NUM_FRAMES, IMG_SIZE, _, _ = model.input_shape

def preprocess_video(video_path):
    """Extract frames and preprocess for model input"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // NUM_FRAMES)
    frames = []

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype("float32") / 255.0
        frames.append(frame)
    cap.release()

    # Pad or truncate to match sequence length
    if len(frames) < NUM_FRAMES:
        frames += [frames[-1]] * (NUM_FRAMES - len(frames))
    else:
        frames = frames[:NUM_FRAMES]

    frames = np.expand_dims(np.array(frames), axis=0)  # (1, T, H, W, C)
    return frames

def predict_video(video_path):
    """Run prediction on a video"""
    frames = preprocess_video(video_path)
    preds = model.predict(frames)
    
    # Handle 1 or 2 output neurons
    if preds.shape[-1] == 1:
        score = preds[0][0]
    else:
        score = preds[0][1]  # assuming [non_violence, violence]

    label = "VIOLENCE" if score > 0.5 else "NON-VIOLENCE"
    print(f"🎥 {os.path.basename(video_path)} → {label} ({score:.3f})")
    return label

# ✅ Test
video_path = "v.mp4"
predict_video(video_path)
