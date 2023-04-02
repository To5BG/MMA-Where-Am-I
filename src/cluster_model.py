from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.cluster import KMeans as kmeans 
import numpy as np

kmeans_model = kmeans(n_clusters=40)
knn_model = knn(n_neighbors=5, weights='distance')

classes = []
entries = []

def reset_models(knn_n=5, kmeans_n=40):
    global knn_model, kmeans_model
    knn_model = knn(n_neighbors=knn_n, weights='distance')
    kmeans_model = kmeans(n_clusters=kmeans_n)

def fit(data):
    sifts = list(map(lambda e: e["sift"], data))
    labels = list(map(lambda e: e["landmark"], data))
    geolocations = list(map(lambda e: [e.get("gps_latitude", 0), e.get("gps_longitude", 0), e.get("gps_altitude", 0)], data))
    fit_features(np.vstack(sifts))
    w = predict_features(np.array(sifts))
    fit_vector(w, labels, geolocations)

def fit_features(data):
    global kmeans_model
    kmeans_model = kmeans_model.fit(data)

def fit_vector(vectors, labels, geos):
    global entries, classes, knn_model
    assert len(vectors) == len(labels)
    indices = []
    for l in labels: 
        if l not in classes: classes.append(l)
        indices.append(classes.index(l))
    classes = np.array(classes)
    knn_model = knn_model.fit(vectors, indices)
    if len(entries) == 0: entries = np.array(geos)
    else: entries = np.concatenate((entries, geos))

def predict(entry):
    w = predict_features([entry[1]["sift"]])
    return predict_vector(w)

def predict_features(data):
    global kmeans_model
    w = np.zeros((len(data), kmeans_model.n_clusters))
    for d in range(len(data)):
        visual_words = kmeans_model.predict(data[d])
        for v in visual_words:
            w[d, v] += 1
    return w

def predict_vector(vector):
    global knn_model, entries, classes
    prediction = classes[knn_model.predict(vector)]
    dist, idx = knn_model.kneighbors(vector)
    weights = 1 / (1e-10 + dist)
    weights /= np.sum(weights, axis=-1)
    geo = []
    for w, g in zip (weights, entries[idx]):
        geo.append(np.matmul(w, g))
    return prediction, np.array(geo)
