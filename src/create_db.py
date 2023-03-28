import pickle
from os import walk
from feature_extraction import get_feature

root_data = "../data"


def walk_folder(path, store):
    for root, dirs, files in walk(path):
        for directory in dirs:
            walk_folder(root + "/" + directory, store)
        for file in files:
            if file.lower().endswith(".jpg"):
                store[root + "/" + file] = {
                    "sift": get_feature(root + "/" + file, "sift"),
                    "landmark": file.split("_")[2],
                    "tag": file.split("_")[-1].split(".")[0]
                }


def create_pickle(destination=root_data, feature="sift"):
    store = {}
    walk_folder(root_data, store)

    # save store as pickle file
    store_file = open(destination + "/store.p", "wb")
    pickle.dump(store, store_file)
    store_file.close()


def read_pickle(destination=root_data):
    store_file = open(destination + "/store.p", "rb")
    store = pickle.load(store_file)
    store_file.close()
    return store


if __name__ == "__main__":
    #create_pickle()
    store = read_pickle()
    print("done")
