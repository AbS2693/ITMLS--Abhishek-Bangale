"""Downloads and verifies the CARLA dataset splits and model weights for this repo.

Setup:
    1. Install the downloader:   pip install gdown
    2. Run this script:          python scripts/download_data.py
       (it downloads each zip below, extracts it into the repo root, then
       deletes the zip; run it from anywhere, paths are resolved relative
       to this file)

Full dataset (all splits + model weights) as one Google Drive folder:
    https://drive.google.com/drive/folders/14l4mbwnsbbGZ0MVNRJN7aeYj6RL5CBuI?usp=drive_link

Alternative: fetch the whole folder directly with gdown instead of running
this script:
    gdown --folder https://drive.google.com/drive/folders/14l4mbwnsbbGZ0MVNRJN7aeYj6RL5CBuI?usp=drive_link -O .

Individual file links (same files as above, listed separately):
    train.zip         https://drive.google.com/file/d/1e9tSHg5W4CVDLEBgx1FpzR2j4Kl6jCqh/view?usp=drive_link
    validation.zip     https://drive.google.com/file/d/1wKPOEDo0-899PiA9v5-cjjnIIHpsT60R/view?usp=drive_link
    test.zip           https://drive.google.com/file/d/1u031oXW9sVkNY9eghr6SvmDEm9JznLya/view?usp=drive_link
    test-fog.zip       https://drive.google.com/file/d/1ZQIPhZrBQS0O0wjPW38bXqr4Pc4hvwUj/view?usp=drive_link
    test-night.zip     https://drive.google.com/file/d/1igjpOghXUIUyrEcAmv4lTkQnPY6-B0eb/view?usp=drive_link
    test-town-01.zip   https://drive.google.com/file/d/1CTAOYOjHFc1qq987mYReW5sRVossICCr/view?usp=drive_link
    Best Models.zip    https://drive.google.com/file/d/1KCmmG2kfeqzPFXN5gCk4PS6rJ5bWvB0T/view?usp=drive_link
"""
import gdown
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Google Drive file IDs (extracted from the links documented above)
DATASET_FILES = {
    "train.zip":        "1e9tSHg5W4CVDLEBgx1FpzR2j4Kl6jCqh",
    "validation.zip":   "1wKPOEDo0-899PiA9v5-cjjnIIHpsT60R",
    "test.zip":         "1u031oXW9sVkNY9eghr6SvmDEm9JznLya",
    "test-fog.zip":     "1ZQIPhZrBQS0O0wjPW38bXqr4Pc4hvwUj",
    "test-night.zip":   "1igjpOghXUIUyrEcAmv4lTkQnPY6-B0eb",
    "test-town-01.zip": "1CTAOYOjHFc1qq987mYReW5sRVossICCr",
    "Best Models.zip":  "1KCmmG2kfeqzPFXN5gCk4PS6rJ5bWvB0T",
}

REQUIRED_DIRS = [
    "train/train/rgb-front", "validation/validation/rgb-front",
    "test/test/rgb-front", "test-fog/test-fog/rgb-front",
    "test-night/test-night/rgb-front", "test-town-01/test-town-01/rgb-front",
    "Best Models",
]

def download_and_extract(filename: str, file_id: str) -> None:
    zip_path = REPO_ROOT / filename
    gdown.download(id=file_id, output=str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(REPO_ROOT)
    zip_path.unlink()

def verify_structure() -> None:
    missing = [d for d in REQUIRED_DIRS if not (REPO_ROOT / d).exists()]
    if missing:
        raise RuntimeError(f"Missing expected directories after download: {missing}")
    print("All expected dataset/model directories are present.")

if __name__ == "__main__":
    for filename, file_id in DATASET_FILES.items():
        print(f"Downloading {filename} ...")
        download_and_extract(filename, file_id)
    verify_structure()
