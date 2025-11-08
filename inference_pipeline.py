import torch
from models.x3d_model import ViolenceX3D
from models.convlstm_autoencoder import ConvLSTMAutoencoder
import cv2
import numpy as np

violence_model = ViolenceX3D()
violence_model.load_state_dict(torch.load("x3d_violence_model.pth"))
violence_model.eval()

anomaly_model = ConvLSTMAutoencoder()
anomaly_model.load_state_dict(torch.load("convlstm_autoencoder.pth"))
anomaly_model.eval()

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.resize(frame, (112, 112)))
    cap.release()

    frames_tensor = torch.tensor(np.array(frames)).permute(0, 3, 1, 2).unsqueeze(0).float()
    violence_pred = torch.softmax(violence_model(frames_tensor), dim=1).argmax().item()
    anomaly_pred = torch.mean((anomaly_model(frames_tensor) - frames_tensor)**2).item()

    print("Violence Detected" if violence_pred else "No Violence")
    print(f"Anomaly Score: {anomaly_pred:.4f}")

if __name__ == "__main__":
    analyze_video("test_video.mp4")
