import create_db
import validation
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='test')
    parser.add_argument('--path', type=str, default='../data/store.p')
    args = parser.parse_args()
    return args

args = get_args()
stage = args.stage
if stage == "generate":
    create_db.create_pickle("../data")
    store = create_db.read_pickle("../data")
    print("Done. Read %d entries." % len(store))
elif stage == "test":
    validation.validate_videos()
elif stage == "predict":
    pass