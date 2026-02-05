import os
import mirdata

# Change this to "saraga_hindustani" for the other branch later
DATASET_NAME = "saraga_carnatic"

def main():
    # Put dataset files inside your repo's data/raw folder
    # (absolute path is safest on Windows)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_home = os.path.join(repo_root, "data", "raw", "mirdata")

    os.environ["MIRDATA_HOME"] = data_home
    print("MIRDATA_HOME =", os.environ["MIRDATA_HOME"])

    ds = mirdata.initialize(DATASET_NAME)

    # Download + validate (first time only; can take a while)
    ds.download()
    ds.validate()

    track_ids = ds.track_ids
    print("Num tracks:", len(track_ids))

    tid = track_ids[0]
    tr = ds.track(tid)

    y, sr = tr.audio  # loads audio into numpy + sample rate
    print("First track:", tid)
    print("Sample rate:", sr)
    print("Duration (s):", len(y) / sr)

if __name__ == "__main__":
    main()
