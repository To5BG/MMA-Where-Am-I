import cv2
import numpy as np

def sift(image, keypoints=10):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(keypoints)
    kp, des = sift.detectAndCompute(gray, None)
    return des