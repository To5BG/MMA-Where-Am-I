import cv2
import create_db
import cluster_model
from exif import Image
import os

#create_db.create_pickle()
store = create_db.read_pickle()
print("Done. Read %d entries." % len(store))

print(cluster_model.fit(list(store.values())[0:10]))
            