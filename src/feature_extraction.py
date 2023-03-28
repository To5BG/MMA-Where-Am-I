import cv2


def sift_features(image, keypoints=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=keypoints)
    kp, des = sift.detectAndCompute(image, None)
    return des


def get_feature(path, feature="sift"):
    image = cv2.imread(path)
    if feature == "sift":
        return sift_features(image)

if __name__ == "__main__":
    print(get_feature("../data/imagefolders/ans/ans_r_nk_p_m_achterkant.jpg").shape)
