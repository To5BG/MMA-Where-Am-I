from create_db import create_pickle,read_pickle
import numpy as np
import cluster_model


def validate(train_test_ratio):
    store = read_pickle("../data")
    train_data = {}
    test_data = {}

    for e in store.items():
        if np.random.random() < train_test_ratio:
            train_data[e[0]] = e[1]
        else:
            test_data[e[0]] = e[1]

    cluster_model.fit(list(train_data.values()))

    acc = 0
    for e in test_data.items():
        if e[1]["landmark"] == cluster_model.predict(e)[0]:
            acc += 1
    print("Accuracy: %f" % (acc / len(test_data)))


if __name__ == "__main__":
    #create_pickle("../data")
    validate(0.7)