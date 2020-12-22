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


class MaskImageDataset(Dataset):
    def __init__(self, path_A, path_B, transforms_=None):
        self.transform_A = transforms.Compose(transforms_[0])
        self.transform_B = transforms.Compose(transforms_[1])
        self.files_A = glob.glob(path_A)
        self.files_A = sorted([path for path in self.files_A if '.jpg' in path])
        self.files_A_mask = [path.split('.jpg')[0] + '.png' for path in self.files_A]
        self.files_B = glob.glob(path_B)
        self.files_B = [img for img in self.files_B if '.png' in img or '.jpg' in img or '.jpeg' in img]

        print('dataset_A: ', path_A, len(self.files_A))
        print('dataset_B: ', path_B, len(self.files_B))

    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]
        path_A_mask = self.files_A_mask[index % len(self.files_A_mask)]
        path_B = self.files_B[randint(0, len(self.files_B) - 1)]

        img_A = np.array(Image.open(path_A).convert('RGB'))
        mask = np.expand_dims(np.array(Image.open(path_A_mask).convert('L')), axis=-1)
        mask[mask > 0] = 1
        img_A = 255*np.ones_like(img_A)*(1-mask) + img_A*mask
        img_A = Image.fromarray(img_A.astype(np.uint8))

        item_A = self.transform_A(img_A)
        item_B = self.transform_B(Image.open(path_B).convert('RGB'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def get_dataloader(pathA, pathB, opt):
    transforms_A = [transforms.RandomResizedCrop(int(opt.size), (0.2, 1.0)),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # (0.5,0.5,0.5) (0.5,0.5,0.5,0.5)
    transforms_B = [transforms.RandomCrop(opt.size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # (0.5,0.5,0.5) (0.5,0.5,0.5,0.5)
    transforms_ls = [transforms_A, transforms_B]
    if opt.mask:
        dataset = MaskImageDataset(opt.path_A, opt.path_B, transforms_=transforms_ls)
    else:
        dataset = ImageDataset(opt.path_A, opt.path_B, transforms_=transforms_ls)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    return dataloader
