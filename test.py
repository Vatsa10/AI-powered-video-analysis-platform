import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

# [LOAD] LOAD SAVEDMODEL
MODEL_PATH = "violence_model_tf"  # folder from model.export()
model = keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serve")

print("[SUCCESS] Model loaded successfully (via TFSMLayer)!")
# No deep introspection needed — the model is ready to call directly.


# [PARAMS] PARAMETERS
NUM_FRAMES = 16
IMG_SIZE = 64
CLASS_NAMES = ["NON-VIOLENCE", "VIOLENCE"]



# [VIDEO] VIDEO PREPROCESSING
def preprocess_video(video_path):
    """
    Extract NUM_FRAMES frames from the video,
    resize, normalize, and return as a batch of shape (1, 16, 64, 64, 3)
    """
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

    # Pad or trim to NUM_FRAMES
    if len(frames) < NUM_FRAMES:
        frames += [frames[-1]] * (NUM_FRAMES - len(frames))
    else:
        frames = frames[:NUM_FRAMES]

    return np.expand_dims(np.array(frames), axis=0)  # shape (1, 16, 64, 64, 3)



# [INFER] INFERENCE FUNCTION
def predict_video(video_path):
    frames = preprocess_video(video_path)
    preds = model(frames, training=False)  # Run inference
    preds = preds.numpy()  # Convert tensor to numpy

    class_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][class_idx])
    label = CLASS_NAMES[class_idx]

    print(f"[VIDEO] {os.path.basename(video_path)} -> {label} ({confidence:.3f})")
    return label, confidence



# [MAIN] MAIN EXECUTION
if __name__ == "__main__":
    video_path = "v.mp4"  # replace with your local test video
    if not os.path.exists(video_path):
        raise FileNotFound(f"[ERROR] Video not found: {video_path}")

    label, confidence = predict_video(video_path)
    print("\n============================")
    print(f"[RESULT] FINAL RESULT: {label}")
    print(f"[SCORE] Confidence: {confidence:.3f}")
    print("============================\n")
