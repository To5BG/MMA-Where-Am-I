from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np

model = knn(n_neighbors=5, weights='distance')

classes = []
entries = []

def reset_model(n=5):
    global model
    model = knn(n_neighbors=n, weights='distance')

def fit(data):
    global entries, classes, model
    for _, v in data:
        if v["landmark"] not in classes:
            classes.append(v["landmark"])
        for d in range(v["sift"].shape[0]):
            model.fit([v["sift"][d]], [classes.index(v["landmark"])])
        entries.append(v)
    entries = np.array(entries)
    
def predict(entry):
    global entries, classes, model
    w = []
    geo = []
    for d in range(entry[1]["sift"].shape[0]):
        prediction = model.predict(entry[1]["sift"][d])
        dist, idx = model.kneighbors(entry[1]["sift"][d])
        weights = 1 / (np.array(dist) * np.sum(dist))
        w.append(prediction)
        geo.append(np.dot(weights, np.array(list(map(lambda i: \
            [entries[i]["gps_latitude"], entries[i]["gps_longitude"], entries[i]["gps_altitude"]], idx)))))
    return w, geo
    
