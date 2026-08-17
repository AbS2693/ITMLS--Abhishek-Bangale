"""Downloads and verifies the CARLA dataset splits and model weights for this repo.
Fill in DATASET_FILES with your own Google Drive file IDs before running.
"""
import gdown
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Google Drive file IDs -- placeholders, replace with your actual shared file IDs
# Clean Google Drive file IDs extracted from your links
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
