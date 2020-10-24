import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, path_A, path_B, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        # self.files_A = glob.glob(os.path.expanduser(os.path.join(path_A + '/*.*')))
        
        self.files_A = glob.glob('/home/freefridays/datasets/256_ObjectCategories/*/**')
        self.files_A = [img for img in self.files_A if '.png' in img or '.jpg' in img or '.jpeg' in img]
        self.files_A = [img for img in self.files_A if len(np.array(Image.open(img)).shape) == 3 and np.array(Image.open(img)).shape[2] == 3]
        
        self.files_B = glob.glob(os.path.expanduser(os.path.join(path_B + '/*.*')))
        self.files_B = [img for img in self.files_B if '.png' in img or '.jpg' in img or '.jpeg' in img]
        self.files_B = [img for img in self.files_B if len(np.array(Image.open(img)).shape) == 3 and np.array(Image.open(img)).shape[2] == 3]


    def __getitem__(self, index):
        path = self.files_A[index % len(self.files_A)]
        item_A = self.transform(Image.open(path))
        

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))