from os import walk
import cv2
from config import config

def get_frames(video_path, freq=1):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % freq == 0:
            frames.append(image)
        success, image = vidcap.read()
        count += 1
    return frames

def read_videos(file_path=config.video_path):
    frames = {}
    for dirpath, _, filenames in walk(file_path):
        dirpath = dirpath.replace("\\", "/")
        for filename in filenames:
            name = dirpath + "/" + filename
            frames[name] = get_frames(name, config.video_sample_rate)
    return frames