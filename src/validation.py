from create_db import read_pickle
import numpy as np
import cluster_model
from video_reader import read_videos
import feature_extraction

avg_geo_loc = [52, 0, 28, 4, 21, 23, 3]

def validate(train_test_ratio):
    store = read_pickle("../data")
    train_data = {}
    test_data = {}

    for e in store.items():
        if np.random.random() < train_test_ratio:
            train_data[e[0]] = e[1]
        else:
            test_data[e[0]] = e[1]

    cluster_model.reset_models()
    cluster_model.fit(list(train_data.values()))

    landmark_acc = 0
    geoloc_mse = []
    for e in test_data.items():
        pred_landmark, pred_geo = cluster_model.predict(e)
        if e[1]["landmark"] == pred_landmark: landmark_acc += 1
        
        ggeo = np.concatenate((e[1]["gps_latitude"], e[1]["gps_longitude"], [e[1]["gps_altitude"]]))
        for i in range(len(ggeo)):
            if ggeo[i] == -1: ggeo[i] = avg_geo_loc[i]
        
        geoloc_mse.append(np.mean(np.square(pred_geo - ggeo) * [3600, 60, 1, 3600, 60, 1, 1]))

    print(np.array(geoloc_mse).shape)
    print("Geolocation weighted MSE (in seconds): %f" % np.mean(geoloc_mse))
    print("Landmark accuracy: %f" % (landmark_acc / len(test_data)))

def predict_videos(file_path):
    store = read_pickle("../data")
    cluster_model.reset_models()
    cluster_model.fit(
        list(filter(lambda e: True if e["landmark"] != "xx" else np.random.random() < 0.25,
                    store.values())))  # reduce the number of xx samples
    frames = read_videos(file_path)

    result = {}
    geo = {}
    for name, video in frames.items():
        result[name] = {
            "oj": 0,
            "nk": 0,
            "rh": 0,
            "xx": 0
        }
        geo[name] = []
        for frame in video:
            ft = (video, {"sift": feature_extraction.sift_features(frame)})
            landmark, geo_loc = cluster_model.predict(ft)
            landmark = landmark[0]

            # for some reason landmark gives a key error in some instances
            if landmark not in result[name]:
                landmark = "xx"
            
            geo[name].append(geo_loc[0])
            result[name][landmark] += 1 / len(video)
        print("Video name: %s" % name)
        print("Geolocation: %s" % np.mean(geo[name], axis=0))
        print(result[name])
        