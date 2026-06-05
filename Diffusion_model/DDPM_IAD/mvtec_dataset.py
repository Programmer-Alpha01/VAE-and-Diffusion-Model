# mvtec_dataset.py  (Fixed test loading)

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

class MVTecDiffusionDataset(Dataset):
    def __init__(self, root_dir, category="bottle", train=True, img_size=256, transform=None):
        self.root = Path(root_dir) / category
        self.train = train
        self.img_size = img_size
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
        ])
        
        if self.train:
            # Only good images for training
            self.img_paths = list((self.root / "train" / "good").glob("*.png"))
        else:
            # Test set: only real images (skip ground_truth folder)
            self.img_paths = []
            test_root = self.root / "test"
            
            for defect_folder in test_root.iterdir():
                if defect_folder.is_dir() and defect_folder.name != "ground_truth":
                    self.img_paths.extend(list(defect_folder.glob("*.png")))
            
            # Optional: sort for reproducibility
            self.img_paths = sorted(self.img_paths)

        print(f"Loaded {len(self.img_paths)} images for {'train' if train else 'test'} set ({category})")

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)