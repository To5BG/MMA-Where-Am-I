import cv2
import numpy as np
from exif import Image

def sift_features(image, keypoints=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=keypoints)
    kp, des = sift.detectAndCompute(image, None)
    return des

def color_hist(im, num_bins=256):
    counts = np.zeros((num_bins, 3))
    bins = np.zeros((num_bins, 3))
    for c in range(0, 3):
        bins[:, c] = np.arange(0, num_bins)
        for i in im[:, :, c].flatten():
            counts[i, c] += 1
    return counts, bins

def get_feature(path, feature="sift"):
    image = cv2.imread(path)
    if feature == "sift":
        return sift_features(image)
    elif feature == "colorhist":
        return color_hist(image)
    
def get_geo_metadata(path):
    with open(path, 'rb') as img_file:
        img = Image(img_file)
        attrs = ['gps_latitude', 'gps_longitude', 'gps_altitude']
        if img.has_exif and all(hasattr(img, s) for s in attrs): 
            return { "gps_latitude": img.gps_latitude, "gps_longitude": img.gps_longitude, "gps_altitude": img.gps_altitude }
        else: return {}