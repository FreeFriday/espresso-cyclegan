import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from torchvision import utils as vutils
# from visdom import Visdom
import numpy as np
from PIL import Image


def save_image(tensor, path, nrow=4):
    grid = vutils.make_grid(tensor.cpu(), nrow=nrow)
    img= (127.5*(grid.float() + 1.0)).permute((1,2,0)).numpy().astype(np.uint8)
    Image.fromarray(img).save(path)


def read_image(path, mode='rgb'):
    return np.array(Image.open(path).convert(mode))


def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))  # 4
    return image.astype(np.uint8)


class Logger:
    def __init__(self, writer):
        self.writer = writer
      
    def log(self, loss_dict, itrs):
        members = [attr for attr in dir(loss_dict)
                  if ((not callable(getattr(loss_dict, attr))
                        and not attr.startswith("__"))
                      and ('loss' in attr
                            or 'grad' in attr
                            or 'nwd' in attr
                            or 'accuracy' in attr))]
        for m in members:
            self.writer.add_scalar(m, getattr(loss_dict, m), itrs + 1)

# class Logger():
#     def __init__(self, n_epochs, batches_epoch):
#         self.viz = Visdom()
#         self.n_epochs = n_epochs
#         self.batches_epoch = batches_epoch
#         self.epoch = 1
#         self.batch = 1
#         self.prev_time = time.time()
#         self.mean_period = 0
#         self.losses = {}
#         self.loss_windows = {}
#         self.image_windows = {}


#     def log(self, losses=None, images=None, print=False):
#         self.mean_period += (time.time() - self.prev_time)
#         self.prev_time = time.time()

#         # sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

#         for i, loss_name in enumerate(losses.keys()):
#             if loss_name not in self.losses:
#                 self.losses[loss_name] = losses[loss_name].data.cpu().numpy()
#             else:
#                 self.losses[loss_name] += losses[loss_name].data.cpu().numpy()

#             # if (i+1) == len(losses.keys()):
#             #     sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
#             # else:
#             #     sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

#         batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
#         batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
#         # sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

#         # End of epoch
#         if (self.batch % self.batches_epoch) == 0:
#             self.print_samples(images)

#             # Plot losses
#             for loss_name, loss in self.losses.items():
#                 if loss_name not in self.loss_windows:
#                     self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
#                                                                     opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
#                 else:
#                     self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
#                 # Reset losses for next epoch
#                 self.losses[loss_name] = 0.0

#             self.epoch += 1
#             self.batch = 1
#             # sys.stdout.write('\n')
#         else:
#             self.batch += 1

#     def print_samples(self, samples):
#         for image_name, tensor in samples.items():
#             if image_name not in self.image_windows:
#                 self.image_windows[image_name] = self.viz.image(tensor2image(tensor),
#                                                                 opts={'title': image_name})
#             else:
#                 self.viz.image(tensor2image(tensor), win=self.image_windows[image_name],
#                                opts={'title': image_name})


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

