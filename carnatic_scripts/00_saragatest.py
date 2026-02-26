import os
import mirdata
# One time only script - Used for checking the computer's access to Saraga and loading inital tracks
# Outputs console print to prove dataset and track list exists, audio loads 

DATASET_NAME = "saraga_carnatic"

def main():
    # Put dataset files inside your repo's data/raw folder
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_home = os.path.join(repo_root, "data", "raw", "mirdata")

    os.environ["MIRDATA_HOME"] = data_home
    print("MIRDATA_HOME =", os.environ["MIRDATA_HOME"])

    # Initialize the dataset
    ds = mirdata.initialize(DATASET_NAME)

    # Download + validate (first time onllyyy - can take a while)
    ds.download()
    ds.validate() # Looks for expected files

    track_ids = ds.track_ids
    print("Num tracks:", len(track_ids))

    tid = track_ids[0] # Loads first audio
    tr = ds.track(tid) # Prints sample rales and duration

    y, sr = tr.audio  # loads audio into numpy + sample rate
    print("First track:", tid)
    print("Sample rate:", sr)
    print("Duration (s):", len(y) / sr)

if __name__ == "__main__":
    main()
