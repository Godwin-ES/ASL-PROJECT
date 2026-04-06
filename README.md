# 🤟 ASL Fingerspelling Recognition & Bidirectional Communication System

This is a real-time **American Sign Language (ASL) fingerspelling recognition system** with **bidirectional communication support** between **signers** (deaf/hard-of-hearing users) and **non-signers** (hearing users).

The system leverages **deep learning, computer vision, and speech processing** to enable communication across language barriers. It was implemented as a **Streamlit web application** for easy accessibility.

---

## 🎯 Objectives

* Develop a CNN-based system to recognize **ASL alphabet letters (A–Z + SPACE)** from webcam input.
* Provide **real-time sign-to-text** and **sign-to-speech** translation for non-signers.
* Implement **speech-to-sign** conversion by recognizing microphone input and displaying corresponding ASL signs.
* Support **fingerspelling**, including repeated letters and spaces for full word construction.

---

## 🖥️ System Features

### 🔹 Signer Mode

* Users sign letters in front of a webcam.
* Model predicts **A–Z + SPACE**.
* Predicted text is displayed and optionally converted to **speech output**.

### 🔹 Non-Signer Mode

* Users speak into the microphone.
* Speech is converted to text.
* Corresponding **ASL alphabet videos** are displayed sequentially.

---

## 📂 Repository Structure

```
ASL-PROJECT/
├── backend/
│   ├── encoder/asl_enc.pkl
│   ├── model/asl_model.pth
│   ├── videos/ (A.mp4, B.mp4, … Z.mp4, SPACE.mp4)
│   ├── model_definition.py
│   ├── sign_recognition.py
│   ├── stt.py
│   ├── tts.py
│   └── video_mapper.py
├── app.py              # Streamlit entry point
├── requirements.txt
└── text.txt
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Godwin-ES/ASL-PROJECT
cd ASL-PROJECT
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📸 Demo Workflow

1. **Signer uploads webcam feed → ASL signs recognized → Text + Speech output.**
2. **Non-signer speaks → Speech recognized → ASL alphabet signs displayed.**

---
