import os
import cv2
import torch
from torch.utils.data import Dataset
from uie.datasets.transforms import get_train_transforms, get_val_transforms


class ImageDataset(Dataset):
    def __init__(self, data_root, mode="train", img_size=224):
        self.root = os.path.join(data_root, mode)
        self.img_size = img_size
        if mode == "train":
            self.transforms = get_train_transforms(img_size)
        else:
            self.transforms = get_val_transforms(img_size)

        self.img_paths = []
        self.labels = []

        classes = sorted(os.listdir(self.root))
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(self.root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for name in os.listdir(cls_dir):
                if name.lower().endswith(('jpg', 'jpeg', 'png')):
                    self.img_paths.append(os.path.join(cls_dir, name))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            augmented = self.transforms(image=img)
            img = augmented["image"]

        return img, torch.tensor(label, dtype=torch.long)