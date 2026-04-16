import os
import csv
import random
import config
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

RANDOM_SEED= 42
MAX_TRANSFORMS_PER_IMAGE= 2
random.seed(RANDOM_SEED)

def horizontal_flip(img):
    return ImageOps.mirror(img)

def vertical_flip(img):
    return ImageOps.flip(img)

def rotate_small(img):
    angle= random.uniform(-12, 12)
    return img.rotate(angle,resample=Image.BICUBIC,expand=False)

def adjust_brightness(img):
    factor= random.uniform(0.8, 1.2)
    return ImageEnhance.Brightness(img).enhance(factor)

def adjust_contrast(img):
    factor= random.uniform(0.8, 1.2)
    return ImageEnhance.Contrast(img).enhance(factor)

def blur_image(img):
    radius= random.uniform(0.3, 1.0)
    return img.filter(ImageFilter.GaussianBlur(radius))

AUGMENTATIONS =[horizontal_flip,vertical_flip,rotate_small,adjust_brightness,adjust_contrast,blur_image]

def apply_random_augmentations(img, max_transforms=2):
    num_transforms= random.randint(1, max_transforms)
    selected_transforms= random.sample(AUGMENTATIONS, k=num_transforms)
    augmented= img.copy()
    for transform in selected_transforms:
        augmented= transform(augmented)
    return augmented

def build_augmented_filename(original_filename):
    stem = Path(original_filename).stem
    return f"{stem}_aug.jpg"

def augment_dataset_and_update_labels(images_root, input_csv, output_csv, percentage):
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader= csv.DictReader(f)
        rows= list(reader)
        required_columns= reader.fieldnames

    output_rows= rows.copy()
    rows=[row for row in rows if row["Subset"]=="train"]
    num_to_augment=int(len(rows)*percentage)
    selected_rows= random.sample(rows, num_to_augment)
    for row in selected_rows:
        original_filename = row["Image file"]
        original_path=os.path.join(images_root, original_filename)

        with Image.open(original_path) as img:
            img = img.convert("RGB")
            augmented_img = apply_random_augmentations(img,max_transforms=MAX_TRANSFORMS_PER_IMAGE)
        new_filename = build_augmented_filename(original_filename)

        new_path = os.path.join(images_root, new_filename)
        augmented_img.save(new_path, format="JPEG", quality=95)
        new_row = row.copy()
        new_row["Image file"]= new_filename
        output_rows.append(new_row)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer= csv.DictWriter(f, fieldnames=required_columns)
        writer.writeheader()
        writer.writerows(output_rows)

if __name__ == "__main__":
    augment_dataset_and_update_labels(images_root=config.TRAIN_PATH,input_csv=config.CSV_PATH,output_csv=config.CSV_AUGMENTED_PATH, percentage=0.15)