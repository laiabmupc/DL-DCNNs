import os
import pandas as pd
from PIL import Image
import config
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(config.CSV_PATH)
correspondance={"train":config.TRAIN_PATH, "val":config.VAL_PATH, "test":config.TEST_PATH}

# Unique classes
unique=df["Medium"].unique()
print(f"Unique classes: {len(unique)}")

# Number of images per subset
for split,path in correspondance.items():
    num_imgs=len(os.listdir(path))
    print(f"{split}: {num_imgs} images")

# Print some examples from the train set
os.makedirs(os.path.join(config.ROOT, "example_images"), exist_ok=True)
train_df=df[df["Subset"]=="train"]
N=5
classes_train=train_df["Medium"].unique()
for cls in classes_train:
    selected=train_df[train_df["Medium"]==cls].head(N)
    fig, axes=plt.subplots(1,N,figsize=(5*N,5), constrained_layout=True)
    fig.suptitle(cls, fontsize=14)
    for ax, (_,row) in zip(axes, selected.iterrows()):
        img_path=os.path.join(config.TRAIN_PATH, row["Image file"])
        try:
            img=Image.open(img_path)
            ax.imshow(img)
            ax.set_axis_off()
        except Exception as e:
            ax.set_axis_off()
            print(f"Error with image {row['Image file']}")
    plt.savefig(os.path.join(config.ROOT, "example_images", f"{cls}.jpg"))
    plt.close()

# Check that all images have the same size
sizes=set()
for _,row in df.iterrows():
    img_path=os.path.join(correspondance[row["Subset"]], row["Image file"])
    try:
        img=Image.open(img_path)
        sizes.add(img.size)
    except Exception as e:
        print(f"Error with image {row['Image file']}")
print(f"Number of different sizes: {len(sizes)}")

# Check class distribution
os.makedirs(os.path.join(config.ROOT, "class_distribution"), exist_ok=True)
for subset in correspondance.keys():
    subset_imgs=df[df["Subset"]==subset]
    counts=subset_imgs["Medium"].value_counts()
    plt.figure()
    counts.plot(kind="bar")
    plt.title(f"{subset} class distribution")
    plt.ylabel("Number of images")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(config.ROOT, "class_distribution", f"{subset}.jpg"))
    plt.close()

# Check intensities and range of intensities per class
os.makedirs(os.path.join(config.ROOT, "class_distribution"), exist_ok=True)
for subset, path in correspondance.items():
    subset_df=df[df["Subset"]==subset]
    files=os.listdir(path)
    pixels_per_class={}
    total_sum=0.0
    total_sum_sq=0.0
    total_count=0
    global_min=255
    global_max=0
    for file in files:
        img_path=os.path.join(path, file)
        cls=subset_df[subset_df["Image file"]==file]["Medium"].values[0]
        try:
            img=Image.open(img_path).convert("RGB")
            arr=np.array(img, dtype=np.uint8)
            flat=arr.flatten()
            if cls not in pixels_per_class.keys():
                pixels_per_class[cls]=[]
            pixels_per_class[cls].append(arr.flatten())
            total_sum +=flat.sum(dtype=np.float64)
            total_sum_sq +=np.square(flat, dtype=np.float64).sum(dtype=np.float64)
            total_count +=flat.size
            global_min=min(global_min, int(flat.min()))
            global_max=max(global_max, int(flat.max()))
        except Exception as e:
            print(f"Error with image {file}")
    for cls in pixels_per_class.keys():
        pixels_per_class[cls]=np.concatenate(pixels_per_class[cls])
    fig, axes=plt.subplots(5,6, figsize=(24,17))
    axes=axes.flatten()
    classes=list(pixels_per_class.keys())
    for i,cls in enumerate(classes):
        axes[i].hist(pixels_per_class[cls], bins=256, range=(0, 255))
        axes[i].set_title(cls, fontsize=10)
        axes[i].set_xlim(0, 255)
        axes[i].tick_params(axis="both", labelsize=7)
    for j in range(len(classes), len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"Histograms: {subset}")
    plt.tight_layout()
    plt.savefig(os.path.join(config.ROOT, "class_distribution", f"intensities_distribution_per_class_{subset}.jpg"))
    plt.close()
    mean=total_sum/total_count
    var=(total_sum_sq/total_count)-(mean**2)
    std=np.sqrt(var)
    print(f"Subset: {subset}")
    print(f"Min: {global_min}")
    print(f"Max: {global_max}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
