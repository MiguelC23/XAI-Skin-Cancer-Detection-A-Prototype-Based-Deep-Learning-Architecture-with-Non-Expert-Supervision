import torch
import numpy as np
import os
import torch.utils.data as data
    
class NpyFolderLoss(data.Dataset):
    """Dataset used for forgetting loss"""

    def __init__(self, loss_directory, device):
        self.loss_directory = loss_directory
        self.device = device
        self.classes, self.class_to_idx = self.find_classes(loss_directory)
        self.samples = self.make_dataset(loss_directory)
        self.pre_loaded_npy, self.pre_loaded_targets = self.load_npy_files()

    def find_classes(self, directory):
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        classes.sort()
        class_to_idx = {cls_name: int(cls_name.split('class_idx_')[-1]) for cls_name in classes}
        return classes, class_to_idx

    def make_dataset(self, directory):
        instances = []
        for target_class in self.class_to_idx.keys():
            class_directory = os.path.join(directory, target_class)
            if not os.path.isdir(class_directory):
                continue

            for npy_file in os.listdir(class_directory):
                if npy_file.endswith('.npy'):
                    path = os.path.join(class_directory, npy_file)
                    item = (path, self.class_to_idx[target_class])
                    instances.append(item)
        return instances

    def load_npy_files(self):
        pre_loaded_npy = []
        pre_loaded_targets = []
        for npy_path, target in self.samples:
            npy_array = np.load(npy_path)
            pre_loaded_npy.append(npy_array)
            pre_loaded_targets.append(target)
        pre_loaded_npy = torch.tensor(np.stack(pre_loaded_npy)).to(self.device)
        pre_loaded_targets = torch.tensor(pre_loaded_targets).to(self.device)
        return pre_loaded_npy, pre_loaded_targets
    
    def __getitem__(self, index):
        npy_path, target = self.samples[index]
        npy_array = np.load(npy_path)
        return npy_array, target

    def __len__(self):
        return len(self.samples)

    def get_all(self):
        return self.pre_loaded_npy, self.pre_loaded_targets
    

import torchvision.transforms as transforms
from PIL import Image
from preprocess import mean, std
from settings import img_size

import re

def get_name_idx_patch(filename):
    # Extrai a parte do nome da string do nome do arquivo
    name_match = re.search(r'(ISIC_\d+)', filename)
    name = name_match.group() if name_match is not None else None
    
    # Extrai a parte do índice do patch da string do nome do arquivo
    idx_patch_match = re.search(r'_(\d+)\.', filename)
    idx_patch = int(idx_patch_match.group(1)) if idx_patch_match is not None else None
    
    return name, idx_patch
    
class ImageFolderLoss(data.Dataset):
    """Dataset used for forgetting loss"""

    def __init__(self, loss_directory, device):
        self.loss_directory = loss_directory
        self.device = device
        self.classes, self.class_to_idx = self.find_classes(loss_directory)
        self.samples = self.make_dataset(loss_directory)
        self.pre_loaded_imgs, self.pre_loaded_targets, self.pre_loaded_names, self.pre_loaded_idx_patches = self.load_images()

    def find_classes(self, directory):
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        classes.sort()
        class_to_idx = {cls_name: int(cls_name.split('class_idx_')[-1]) for cls_name in classes}
        return classes, class_to_idx

    def make_dataset(self, directory):
        instances = []
        for target_class in self.class_to_idx.keys():
            class_directory = os.path.join(directory, target_class)
            if not os.path.isdir(class_directory):
                continue

            for img_file in os.listdir(class_directory):
                if img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png'):
                    path = os.path.join(class_directory, img_file)
                    name, idx_patch = get_name_idx_patch(img_file)
                    item = (path, self.class_to_idx[target_class], name, idx_patch)
                    instances.append(item)
        return instances

    def load_images(self):
        pre_loaded_imgs = []
        pre_loaded_targets = []
        pre_loaded_names = []
        pre_loaded_idx_patches = []
        for img_path, target, name, idx_patch in self.samples:
            img = Image.open(img_path)
            img = img.convert('RGB')
            img_tensor = self.transforms(img)
            pre_loaded_imgs.append(img_tensor)
            pre_loaded_targets.append(target)
            pre_loaded_names.append(name)
            pre_loaded_idx_patches.append(idx_patch)
        pre_loaded_imgs = torch.stack(pre_loaded_imgs).to(self.device)
        pre_loaded_targets = torch.tensor(pre_loaded_targets).to(self.device)
        pre_loaded_idx_patches = torch.tensor(pre_loaded_idx_patches).to(self.device)
        return pre_loaded_imgs, pre_loaded_targets, pre_loaded_names, pre_loaded_idx_patches

    @property
    def transforms(self):
        return transforms.Compose([
            transforms.Resize(size=(img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def __getitem__(self, index):
        img_path, target, name, idx_patch = self.samples[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img_tensor = self.transforms(img)
        return img_tensor, target, name, idx_patch

    def __len__(self):
        return len(self.samples)

    def get_all(self):
        return self.pre_loaded_imgs, self.pre_loaded_targets, self.pre_loaded_names, self.pre_loaded_idx_patches


    




