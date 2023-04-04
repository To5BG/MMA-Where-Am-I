import cv2
import numpy as np
from exif import Image
from config import config

def sift_features(image, keypoints=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #resize image to resize_width
    ratio = config.resize_width / image.shape[1]
    dim = (int(image.shape[0] * ratio), config.resize_width)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    sift = cv2.SIFT_create(nfeatures=keypoints)
    _, des = sift.detectAndCompute(image, None)
    return des[:keypoints, :]

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
        if not img.has_exif: return {"gps_latitude": [-1, -1, -1], "gps_longitude": [-1, -1, -1], "gps_altitude": -1}
        geo = {}
        for s, defo in zip(attrs, [[-1, -1, -1], [-1, -1, -1], -1]): 
            geo[s] = getattr(img, s) if hasattr(s, img) else defo
        return geo 