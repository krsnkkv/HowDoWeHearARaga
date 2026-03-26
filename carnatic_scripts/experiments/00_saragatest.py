"""
00_saragatest.py — Environment Validation (One-Time Setup)

WHAT THIS SCRIPT DOES:
    One-time setup script that verifies the computer has access to the Saraga
    Carnatic dataset via mirdata. Downloads and validates the dataset if not
    already present, loads the first track, and prints its sample rate and
    duration to confirm audio loading works correctly.

LOGIC AND DESIGN:
    Sets MIRDATA_HOME to data/raw/mirdata within the repo, initialises the
    saraga_carnatic dataset, calls download() and validate() to ensure all
    expected files are present. Loads the first track's audio into a numpy array
    as a smoke test. This script is run once at project setup and never again.

COGNITIVE / MUSICOLOGICAL RATIONALE:
    No cognitive component — this is purely infrastructure. However, the Saraga
    Carnatic dataset was chosen over alternatives (e.g., RagaDataset, CompMusic
    corpora) because it provides raga labels, ground-truth tonic annotations,
    and full audio — all three are required for the pipeline's interpretable
    feature extraction approach.
"""
import os
import mirdata

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
