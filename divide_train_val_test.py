import os
import shutil
import config
import pandas as pd

df=pd.read_csv(config.CSV_PATH)
os.makedirs(config.TRAIN_PATH, exist_ok=True)
os.makedirs(config.TEST_PATH, exist_ok=True)
os.makedirs(config.VAL_PATH, exist_ok=True)
correspondance={"train":config.TRAIN_PATH, "val":config.VAL_PATH, "test":config.TEST_PATH}

for _,row in df.iterrows():
    filename=row["Image file"]
    subset=row["Subset"]
    src=os.path.join(config.DATA, filename)
    destination=os.path.join(correspondance[subset], filename)

    if os.path.exists(src):
        shutil.move(src, destination)
    else:
        print(f"Not found {filename}")
print("Dataset divided")