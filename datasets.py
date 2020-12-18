import glob
from random import randint
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, path_A, path_B, transforms_=None):
        self.transform_A = transforms.Compose(transforms_[0])
        self.transform_B = transforms.Compose(transforms_[1])
        self.files_A = glob.glob(path_A)
        self.files_A = [img for img in self.files_A if '.png' in img or '.jpg' in img or '.jpeg' in img]
        self.files_B = glob.glob(path_B)
        self.files_B = [img for img in self.files_B if '.png' in img or '.jpg' in img or '.jpeg' in img]

        print('dataset_A: ', len(self.files_A))
        print('dataset_B: ', len(self.files_B))

    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]
        path_B = self.files_B[randint(0, len(self.files_B) - 1)]
        item_A = self.transform_A(Image.open(path_A).convert('RGB'))
        item_B = self.transform_B(Image.open(path_B).convert('RGB'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def get_dataloader(pathA, pathB, opt):
    transforms_A = [transforms.RandomResizedCrop(int(opt.size), (0.2, 1.0)),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # (0.5,0.5,0.5) (0.5,0.5,0.5,0.5)
    transforms_B = [transforms.Resize(int(opt.size), Image.BOX),  # Image.BICUBIC
                    transforms.RandomCrop(opt.size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # (0.5,0.5,0.5) (0.5,0.5,0.5,0.5)
    transforms_ls = [transforms_A, transforms_B]
    dataloader = DataLoader(ImageDataset(opt.path_A, opt.path_B, transforms_=transforms_ls),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    return dataloader
