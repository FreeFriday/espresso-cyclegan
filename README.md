# Pytorch-CycleGAN
A clean and readable Pytorch implementation of CycleGAN (https://arxiv.org/abs/1703.10593)

## Prerequisites
Code is intended to work with ```Python 3.6.x```, it hasn't been tested with previous versions

### [PyTorch & torchvision](http://pytorch.org/)
Follow the instructions in [pytorch.org](http://pytorch.org) for your current setup

### [Visdom](https://github.com/facebookresearch/visdom)
To plot loss graphs and draw images in a nice web browser view
`python train.py --path_A=/home/freefridays/datasets/photo2som/trainA/* --path_B=/home/freefridays/datasets/photo2som/trainB/* --batch_size=3 --name=my_dir_name`

`python inference.py --input=input/path.png --output=output/path.png --output_inout=output_with_input/path.png`