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
        model.fit([v["sift"].flatten()], [classes.index(v["landmark"])])
        entries.append(v)
    entries = np.array(entries)
    
def predict(entry):
    global entries, classes, model
    prediction = model.predict(entry["sift"])
    dist, idx = model.kneighbors(entry["sift"])

    weights = 1 / (np.array(dist) * np.sum(dist))
    near_geo = np.array(list(map(lambda i: \
            [entries[i]["gps_latitude"], entries[i]["gps_longitude"], entries[i]["gps_altitude"]], idx)))
    return prediction, np.dot(weights, near_geo)
    
