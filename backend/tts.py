from gtts import gTTS
import io

def text_to_speech(text: str, lang: str = 'en'):
    if not text:
        return None

    try:
        # Generate speech
        tts = gTTS(text=text, lang=lang)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        
        return audio_fp
    except Exception as e:
        print(f"TTS Error: {e}")
        return None
