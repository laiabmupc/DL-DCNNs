import os
import config
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MAMeDataset(Dataset):
    def __init__(self, root_dir=config.ROOT, transform=None, path_csv=config.CSV_AUGMENTED_PATH, classes_to_ind=None, num_imgs=0, data_aug=False):
        self.root_dir = root_dir
        self.transform=transform
        self.image_paths=sorted(os.listdir(root_dir))
        self.data_aug=data_aug
        self.path_csv= path_csv
        if not data_aug:
            self.image_paths=[path for path in self.image_paths if "_aug" not in path]
            path_csv=config.CSV_PATH
        if num_imgs>0:
            self.image_paths=self.image_paths[:num_imgs]
        self.metadata = pd.read_csv(path_csv)
        if classes_to_ind is None:
            self.classes=sorted(self.metadata["Medium"].unique())
            self.classes_to_ind={cls: i for i, cls in enumerate(self.classes)}
        else:
            self.classes_to_ind=classes_to_ind

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_paths[idx])
        img=Image.open(image_path).convert("RGB")
        if self.transform is not None:
            img=self.transform(img)
        label=self.metadata[self.metadata["Image file"]==self.image_paths[idx]]["Medium"].values[0]
        label_id = self.classes_to_ind[label]
        return img, label_id, idx