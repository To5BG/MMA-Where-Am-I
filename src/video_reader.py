from os import walk
import cv2
import numpy as np

video_path = "../data/imagefolders/test_videos"

def get_frames(video_path, freq=1):
    frames = np.array([])
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % freq == 0:
            frames = np.append(frames, image)
            #cv2.imshow("frames", image)
            #cv2.waitKey(1)
        success, image = vidcap.read()
        count += 1
    return frames

def read_videos():
    for dirpath, dirnames, filenames in walk(video_path):
        for (i, filename) in enumerate(filenames):
            get_frames(video_path + "/" + filename, 4)

read_videos()