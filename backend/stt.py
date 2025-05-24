# backend/stt.py

import io
#from pydub import AudioSegment
import speech_recognition as sr

def speech_to_text(audio_bytes: bytes) -> str:
    try:
        recognizer = sr.Recognizer()
        audio_file = io.BytesIO(audio_bytes)

        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        return text.upper()
    except Exception as e:
        print(f"[STT ERROR] {e}")
        return f"[STT ERROR] {e}"    
    '''
    try:
        # Convert to WAV using pydub
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        # Transcribe with speech_recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        return text.upper()
    except Exception as e:
        print(f"[STT ERROR] {e}")
        return f"[STT ERROR] {e}"
    '''