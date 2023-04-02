import pickle
import os
import feature_extraction
from config import config


def walk_folder(path, store):
    for root, dirs, files in os.walk(path):
        root = root.replace("\\", "/")
        for file in files:
            if file.lower().endswith(".jpg"):
                print("Processing %s" % (root + "/" + file))
                store[root + "/" + file] = {
                    "sift": feature_extraction.get_feature(root + "/" + file, "sift"),
                    "landmark": file.split("_")[2],
                    "tag": file.split("_")[-1].split(".")[0],
                } | feature_extraction.get_geo_metadata(root + "/" + file)


def create_pickle(destination, feature="sift"):
    store = {}
    walk_folder(config.root_data, store)

    # save store as pickle file
    store_file = open(destination + "/store.p", "wb")
    pickle.dump(store, store_file)
    store_file.close()


def read_pickle(destination):
    store_file = open(destination + "/store.p", "rb")
    store = pickle.load(store_file)
    store_file.close()
    return store
