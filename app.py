import os
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras

# -------------------------------
# ✅ Load TensorFlow SavedModel (Keras 3)
# -------------------------------
MODEL_PATH = "violence_model_tf"
model = keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serve")

NUM_FRAMES = 16
IMG_SIZE = 64
CLASS_NAMES = ["NON-VIOLENCE", "VIOLENCE"]

# -------------------------------
# Flask setup
# -------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['FRAME_FOLDER'] = 'static/frames'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FRAME_FOLDER'], exist_ok=True)

# -------------------------------
# 🎥 Preprocess for overall prediction
# -------------------------------
def preprocess_video(video_path):
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

    if len(frames) < NUM_FRAMES:
        frames += [frames[-1]] * (NUM_FRAMES - len(frames))
    else:
        frames = frames[:NUM_FRAMES]

    return np.expand_dims(np.array(frames), axis=0)

# -------------------------------
# 🔍 Frame-by-Frame Analysis (with thumbnails)
# -------------------------------
def analyze_video_frames(video_path, video_name):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // NUM_FRAMES)
    frame_results = []

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame preview
        frame_filename = f"{video_name}_frame_{i}.jpg"
        frame_path = os.path.join(app.config['FRAME_FOLDER'], frame_filename)
        frame_small = cv2.resize(frame, (256, 144))
        cv2.imwrite(frame_path, frame_small)

        # Prepare for inference
        processed = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        processed = processed.astype("float32") / 255.0

        # Duplicate to shape (1,16,64,64,3)
        clip = np.repeat(np.expand_dims(processed, axis=0), NUM_FRAMES, axis=0)
        clip = np.expand_dims(clip, axis=0)

        # Predict
        pred = model(clip, training=False).numpy()
        label_idx = int(np.argmax(pred[0]))
        label = CLASS_NAMES[label_idx]
        conf = float(pred[0][label_idx]) * 100

        frame_results.append({
            "frame_img": f"/{frame_path}",
            "label": label,
            "confidence": f"{conf:.2f}%"
        })

    cap.release()
    return frame_results

# -------------------------------
# 🌐 Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)
    video_name = os.path.splitext(file.filename)[0]

    # Overall video prediction
    frames = preprocess_video(video_path)
    preds = model(frames, training=False).numpy()
    class_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][class_idx]) * 100
    label = CLASS_NAMES[class_idx]

    # Frame-by-frame analysis
    frame_results = analyze_video_frames(video_path, video_name)

    return jsonify({
        "video_path": f"/{video_path}",
        "overall_label": label,
        "overall_confidence": f"{confidence:.2f}%",
        "frame_results": frame_results
    })


if __name__ == '__main__':
    app.run(debug=True)
