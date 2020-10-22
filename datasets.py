import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root='', transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join('', '/content/drive/My Drive/Crawling/coil-100') + '/*.*'))
        self.files_A = [img for img in self.files_A if '.png' in img or '.jpg' in img or '.jpeg' in img]
        self.files_B = sorted(glob.glob(os.path.join('', '/content/drive/My Drive/Crawling/cropped') + '/*.*'))
        self.files_B = [img for img in self.files_B if '.png' in img or '.jpg' in img or '.jpeg' in img]


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))