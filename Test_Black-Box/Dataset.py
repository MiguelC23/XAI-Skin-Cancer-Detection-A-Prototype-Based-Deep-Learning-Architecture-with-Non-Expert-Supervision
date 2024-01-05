import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from preprocess import mean, std
img_size = 224

def replace_values(x):
    return torch.where(x < 0.5, torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32))

class ISIC2019_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, is_train=True, is_push=False,number_classes=0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        if(number_classes==8):
            self.classes = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
        elif number_classes==2:
            self.classes = ['MEL', 'NV']
        else :
            print("Number of classes can only be 8 or 2\n")
            print("Error...Exit\n")
            exit(0)

        self.mask_transform =transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(replace_values)
        ])

        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        if is_push:
            self.transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
            ])

        self.ids = []
        for label in os.listdir(self.image_dir):
            if label in  self.classes:
                label_dir = os.path.join(self.image_dir, label)
                label_idx = self.classes.index(label)
                for img_name in os.listdir(label_dir):
                    img_id = img_name[:12]
                    img_path = os.path.join(label_dir, img_name)
                    if self.mask_dir is not None:
                        mask_path = os.path.join(self.mask_dir, label, f"{img_id}.png")
                    else:
                        mask_path = None
                    self.ids.append((img_path, label_idx, img_id, mask_path))

    def __getitem__(self, index):
        img_path, label, img_id, mask_path = self.ids[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label is not None:
            if self.mask_dir is not None and mask_path is not None:
                mask = Image.open(mask_path).convert('L')
                mask = self.mask_transform(mask)
            else:
                mask = torch.zeros((1, img.shape[1], img.shape[2]))
            return img, label, img_id, mask
        else:
            return img, label, img_id, mask

    def __len__(self):
        return len(self.ids)


class PH2_Dataset_or_Derm7pt(Dataset):
    def __init__(self, image_dir, mask_dir=None, is_train=True, is_push=False,number_classes=0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        if(number_classes==8):
            self.classes = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
        elif number_classes==2:
            self.classes = ['MEL', 'NV']
        else :
            print("Number of classes can only be 8 or 2\n")
            print("Error...Exit\n")
            exit(0)

        self.mask_transform =transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(replace_values)
        ])

        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        if is_push:
            self.transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
            ])

        self.ids = []
        for label in os.listdir(self.image_dir):
            if label in  self.classes:
                label_dir = os.path.join(self.image_dir, label)
                label_idx = self.classes.index(label)
                for img_name in os.listdir(label_dir):
                    img_id = os.path.splitext(img_name)[0]
                    img_path = os.path.join(label_dir, img_name)
                    if self.mask_dir is not None:
                        mask_path = os.path.join(self.mask_dir, label, f"{img_id}.png")
                    else:
                        mask_path = None
                    self.ids.append((img_path, label_idx, img_id, mask_path))

    def __getitem__(self, index):
        img_path, label, img_id, mask_path = self.ids[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label is not None:
            if self.mask_dir is not None and mask_path is not None:
                mask = Image.open(mask_path).convert('L')
                mask = self.mask_transform(mask)
            else:
                mask = torch.zeros((1, img.shape[1], img.shape[2]))
            return img, label, img_id, mask
        else:
            return img, label, img_id, mask

    def __len__(self):
        return len(self.ids)
    


class PH2_Dataset_or_Derm7pt_XAI(Dataset):
    def __init__(self, image_dir, mask_dir=None, is_train=True, is_push=False,number_classes=0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        if(number_classes==8):
            self.classes = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
        elif number_classes==2:
            self.classes = ['MEL', 'NV']
        else :
            print("Number of classes can only be 8 or 2\n")
            print("Error...Exit\n")
            exit(0)

        self.mask_transform =transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(replace_values)
        ])

        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=mean, std=std)
        ])

        if is_push:
            self.transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
            ])

        self.ids = []
        for label in os.listdir(self.image_dir):
            if label in  self.classes:
                label_dir = os.path.join(self.image_dir, label)
                label_idx = self.classes.index(label)
                for img_name in os.listdir(label_dir):
                    img_id = os.path.splitext(img_name)[0]
                    img_path = os.path.join(label_dir, img_name)
                    if self.mask_dir is not None:
                        mask_path = os.path.join(self.mask_dir, label, f"{img_id}.png")
                    else:
                        mask_path = None
                    self.ids.append((img_path, label_idx, img_id, mask_path))

    def __getitem__(self, index):
        img_path, label, img_id, mask_path = self.ids[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label is not None:
            if self.mask_dir is not None and mask_path is not None:
                mask = Image.open(mask_path).convert('L')
                mask = self.mask_transform(mask)
            else:
                mask = torch.zeros((1, img.shape[1], img.shape[2]))
            return img, label, img_id, mask
        else:
            return img, label, img_id, mask

    def __len__(self):
        return len(self.ids)