# backend/stt.py

import io
import speech_recognition as sr

def speech_to_text(audio_bytes: bytes) -> str:
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data).upper()
    except Exception as e:
        print(f"[STT ERROR] {e}")
        return f""