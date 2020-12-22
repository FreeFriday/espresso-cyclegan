import argparse
import os

# arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='input image path')
parser.add_argument('--output', type=str, default='./result.png', required=False, help='output image path')
parser.add_argument('--output_inout', type=str, default='./result_inout.png', required=False, help='output image path')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--model', type=str, required=True, help='model path')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--device', type=str, default='0', help='GPU ID')
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
print(opt)

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from models import Generator
from utils import save_image


# model
netG_A2B = Generator(opt.input_nc, opt.output_nc).cuda()
netG_B2A = Generator(opt.output_nc, opt.input_nc).cuda()
state = torch.load(opt.model)
netG_A2B.load_state_dict(state['netG_A2B'])
netG_B2A.load_state_dict(state['netG_B2A'])


def pad_factor(img, factor=16):
    W, H = img.size
    pad_W = (factor - (W % factor)) % factor
    pad_H = (factor - (H % factor)) % factor
    padding = ((pad_H // 2, pad_H - (pad_H // 2)), (pad_W // 2, pad_W - (pad_W // 2)), (0, 0))

    img_np = np.array(img)
    img_pad = np.pad(img_np, padding, 'reflect')
    img_pad = Image.fromarray(img_pad)

    return img_pad, pad_W, pad_H


def original_region(img, pad_H, pad_W):
    h0 = pad_H // 2
    h1 = -(pad_H - (pad_H // 2)) if pad_H != 0 else None
    w0 = pad_W // 2
    w1 = -(pad_W - (pad_W // 2)) if pad_W != 0 else None
    return img[:, :, h0:h1, w0:w1]


def transform_sample(sample):
    transforms_sample_ = [ transforms.ToTensor(),
                           transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    for transform in transforms_sample_:
        sample = transform(sample)
    sample = torch.unsqueeze(sample, 0)
    return sample


def inference():
    _input = Image.open(opt.input).convert('RGB')
    _input_pad, pad_W, pad_H = pad_factor(_input)
    _input_pad = transform_sample(_input_pad).cuda()
    _output = netG_A2B(_input_pad)
    _output = original_region(_output, pad_H, pad_W)
    result = torch.cat([transform_sample(_input).detach().cpu(), _output.detach().cpu()], dim=0)
    save_image(result, opt.output_inout, nrow=1)
    save_image( _output.detach().cpu(), opt.output, nrow=1)


if __name__ == '__main__':
    inference()
