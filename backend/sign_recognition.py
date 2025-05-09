# backend/sign_recognition.py

import torch
import joblib
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from cvzone.HandTrackingModule import HandDetector

from backend.model_definition import ASLModel
    
# === Load model, encoder, transforms ===
model_path = r"backend\model\asl_model.pth"
encoder_path = r"backend\encoder\asl_enc.pkl"

asl_model = ASLModel(num_classes=26, model_name='efficientnet_b0')
asl_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
asl_model.eval()

le = joblib.load(encoder_path)

detector = HandDetector(detectionCon=0.8)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_sign(image):
    if image is None:
        print("No image received.")
        return "No Image"

    hands, image = detector.findHands(image, draw=False, flipType=True)

    if not hands:
        #print("No hand detected.")
        return "No Hand Detected"

    hand = hands[0]
    x, y, w, h = hand['bbox']

    # Padding and bounds
    pad = 20
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, image.shape[1])
    y2 = min(y + h + pad, image.shape[0])

    hand_crop = image[y1:y2, x1:x2]
    hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(hand_rgb)

    img_tensor = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        logits = asl_model(img_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1)
        label = le.inverse_transform(pred_idx.numpy())[0]
    return label
