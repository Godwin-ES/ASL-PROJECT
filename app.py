import streamlit as st
import av
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
from st_audiorec import st_audiorec
from streamlit_webrtc import webrtc_streamer

from backend.sign_recognition import predict_sign
from backend.tts import text_to_speech
from backend.stt import speech_to_text
from backend.video_mapper import get_asl_video

st.set_page_config(page_title="ASL Communication Portal")
st.title("ASL Communication Portal")

user_type = st.radio("Choose your role:", ("Signer", "Non-Signer"))

previous_letter = ""
text_path = "text.txt"

# --- Signer Section ---
if user_type == "Signer":
    st.subheader("ASL Signer Mode")
    result_container = st.empty()
    audio_container = st.empty()
    
    def video_frame_callback(frame):
        global previous_letter
        img = frame.to_ndarray(format="bgr24")
        try:
            predicted_letter = predict_sign(img)

            # Only generate and save audio if a valid letter
            if predicted_letter.isalpha():
                cv2.putText(img, predicted_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if predicted_letter != previous_letter:
                    with open(text_path, "w") as file:
                        file.write(predicted_letter)
                    previous_letter = predicted_letter
                
            else:
                cv2.putText(img, predicted_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)              

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
                
    if webrtc_ctx and not webrtc_ctx.state.playing:
        previous_letter = ''
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

    