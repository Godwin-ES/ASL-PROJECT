# backend/video_mapper.py

import os
from typing import List, Tuple

ASL_VIDEO_DIR = r"backend\videos"

def get_asl_video(text: str) -> List[Tuple[str, List[str]]]:
    result = []

    words = text.upper().split()
    for word in words:
        letter_videos = []
        for letter in word:
            if letter.isalpha():
                video_path = os.path.join(ASL_VIDEO_DIR, f"{letter}.mp4")
                if os.path.exists(video_path):
                    letter_videos.append(video_path)
                else:
                    print(f"[WARNING] Video for '{letter}' not found.")
        if letter_videos:
            result.append((word, letter_videos))

    return result