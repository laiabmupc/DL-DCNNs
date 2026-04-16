from pathlib import Path
import torch
import os
ROOT=Path('/home/nct/nct01188/DL-lab')
DATA=ROOT / "data"
TRAIN_PATH=ROOT / "train"
VAL_PATH=ROOT / "val"
TEST_PATH=ROOT / "test"
CSV_PATH=ROOT / "MAMe_dataset.csv"
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
RESULTS=ROOT / "results"
os.makedirs(RESULTS, exist_ok=True)
CSV_AUGMENTED_PATH=ROOT / "MAMe_dataset_aug.csv"