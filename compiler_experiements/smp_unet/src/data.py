# dataset.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import numpy as np
from PIL import Image

def get_transforms(image_size=(256,256)):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size, interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size=(256,256)):
        self.ds = dataset
        self.img_size = img_size
        self.img_transform = get_transforms(img_size)
        self.mask_transform = get_transforms(img_size)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        row = self.ds[idx]
        img = np.array(row["image"])
        mask = np.array(row["mask"])
        
        img = self.img_transform(img)  # C×H×W floats in [0,1]
        mask = self.mask_transform(mask)  # 1×H×W floats
        mask = mask * 255  # bring to [0,255]
        mask = mask.long().squeeze(0)  # H×W ints with class indices
        
        return img, mask

def get_dataloaders(batch_size=8, img_size=(256,256), num_workers=4):
    # Split the HF dataset into train/val/test
    full = load_dataset("mattmdjaga/human_parsing_dataset", split="train")
    counts = full.train_test_split(test_size=0.2, seed=42)
    train_valid = counts["train"]
    test = counts["test"]
    tv = train_valid.train_test_split(test_size=0.1, seed=42)
    train = tv["train"]
    val = tv["test"]
    
    # Create datasets directly from the split datasets
    train_ds = HFDataset(dataset=train, img_size=img_size)
    val_ds = HFDataset(dataset=val, img_size=img_size)
    test_ds = HFDataset(dataset=test, img_size=img_size)
    
    loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, **loader_args)
    val_loader = DataLoader(val_ds, **loader_args)
    test_loader = DataLoader(test_ds, **loader_args)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_dl, val_dl, test_dl = get_dataloaders()
    img, mask = next(iter(train_dl))
    print("Batch shapes:", img.shape, mask.shape)