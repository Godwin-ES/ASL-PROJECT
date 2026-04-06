import streamlit as st
import av
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
from datetime import datetime
from st_audiorec import st_audiorec
from streamlit_webrtc import webrtc_streamer

from backend.sign_recognition import predict_sign
from backend.tts import text_to_speech
from backend.stt import speech_to_text
from backend.video_mapper import get_asl_video

st.set_page_config(page_title="ASL Communication System")
st.title("ASL Communication System")

user_type = st.radio("Choose your role:", ("Signer", "Non-Signer"))

sentence = ""
last_prediction = ""
last_update_time = datetime.now()
space_start_time = None
prediction_history = []
text_path = "text.txt"

# Inside the Signer section
if user_type == "Signer":
    st.subheader("ASL Signer Mode")
    result_container = st.empty()
    audio_container = st.empty()
    
    def video_frame_callback(frame):
        global last_prediction, last_update_time, sentence, space_start_time, prediction_history

        img = frame.to_ndarray(format="bgr24")
        try:
            predicted_letter = predict_sign(img)
            current_time = datetime.now()
            
            # --- Simple prediction smoothing: Use mode of last 10 predictions ---
            prediction_history.append(predicted_letter)
            if len(prediction_history) > 10:
                prediction_history.pop(0)
            smoothed_letter = max(set(prediction_history), key=prediction_history.count)

            if smoothed_letter == last_prediction:
                # If current prediction is the same as the last, track SPACE time
                if smoothed_letter == "SPACE":
                    if space_start_time is None:
                        space_start_time = current_time
                    elapsed = (current_time - space_start_time).total_seconds()
                    
                    if elapsed >= 3.0:
                        if not sentence.endswith(" "):
                            sentence += " "
                        with open(text_path, "w") as f:
                            f.write(sentence)
            else:
                # New letter prediction
                if smoothed_letter.isalpha():
                    sentence += smoothed_letter
                    with open(text_path, "w") as f:
                        f.write(sentence)
                    space_start_time = None  # Reset space timer

                last_prediction = smoothed_letter
                last_update_time = current_time

            # Draw prediction on frame
            cv2.putText(img, smoothed_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Show running sentence
            cv2.putText(img, f"Sentence: {sentence}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        except Exception as e:
            st.warning(f"Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="signer-mode",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
    
        # Play audio on button press
    if st.button("🔊 Play Audio"):
        if os.path.exists(text_path):
            with open(text_path, "r") as file:
                text = file.read()
            if text:
                audio_fp = text_to_speech(text)
                audio_fp.seek(0)  # Reset pointer to the beginning
                st.audio(audio_fp, format="audio/mpeg")
                
    if (webrtc_ctx and not webrtc_ctx.state.playing) or st.button("Clear Sentence"):
        last_prediction = ""
        sentence = ""
        space_start_time = None
        if os.path.exists(text_path):
            with open(text_path, "w") as f:
                pass
                    
# --- Non-Signer Section ---
else:
    st.subheader("Non-Signer Mode")
    st.info("Click to record your voice. The app will transcribe and show corresponding ASL signs.")

    audio = st_audiorec()
        
    if audio:
        # Transcribe speech using backend STT
        with st.spinner("Transcribing..."):
            recognized_text = speech_to_text(audio)

        if not recognized_text.strip():
            st.warning("No speech recognized. Please try again.")
        else:
            st.success(f"You said: {recognized_text}")

            # Map text to ASL videos
            video_mappings = get_asl_video(recognized_text)

            if not video_mappings:
                st.warning("No matching ASL videos found.")
            else:
                st.subheader("ASL Video Representation:")
                for word, videos in video_mappings:
                    st.markdown(f"**{word}**")
                    cols = st.columns(len(videos))
                    for col, video in zip(cols, videos):
                        with col:
                            st.video(video)

    