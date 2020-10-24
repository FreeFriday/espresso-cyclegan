import argparse
import itertools
import os
from tqdm import tqdm
from datetime import datetime

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import save_image
from utils import weights_init_normal
from utils import read_image
from datasets import ImageDataset


# arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='input image path')
parser.add_argument('--output', type=str, default='./result.jpg', required=False, help='output image path')
parser.add_argument('--model', type=str, required=True, help='model path')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
opt = parser.parse_args()

# model
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
state = torch.load(opt.model)
netG_A2B.load_state_dict(state['netG_A2B'])
netG_B2A.load_state_dict(state['netG_B2A'])


def transform_sample(sample):
    transforms_sample_ = [ transforms.ToTensor(),
                           transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    for transform in transforms_sample_:
        sample = transform(sample)
    sample = torch.unsqueeze(sample, 0)
    return sample


def inference():
    _input = transform_sample(read_image(opt.input)).cuda()
    _output = netG_A2B(_input)
    result = torch.cat([_input.detach(), _output.detach()], dim=0)
    save_image(result, opt.output, nrow=1)


if __name__ == '__main__':
    inference()
