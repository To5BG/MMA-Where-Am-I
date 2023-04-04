from create_db import read_pickle
import numpy as np
import cluster_model
from video_reader import read_videos
import feature_extraction

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

    acc = 0
    for e in test_data.items():
        if e[1]["landmark"] == cluster_model.predict(e)[0]:
            acc += 1
    print("Accuracy: %f" % (acc / len(test_data)))

def predict_videos(file_path):
    store = read_pickle("../data")
    cluster_model.reset_models()
    cluster_model.fit(
        list(filter(lambda e: True if e["landmark"] != "xx" else np.random.random() < 0.1,
                    store.values())))  # reduce the number of xx samples
    frames = read_videos(file_path)

    result = {}
    for name, video in frames.items():
        result[name] = {
            "oj": 0,
            "nk": 0,
            "rh": 0,
            "xx": 0
        }
        for frame in video:
            ft = (video, {"sift": feature_extraction.sift_features(frame)})
            landmark = cluster_model.predict(ft)[0][0]

            # for some reason landmark gives a key error in some instances
            if landmark not in result[name]:
                landmark = "xx"

            result[name][landmark] += 1 / len(video)
        print("Video name: %s" % name)
        print(result[name])
        