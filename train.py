import argparse
import itertools
import os
from tqdm import tqdm
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--path_A', type=str, help='directory path of domain A', default='/home/freefridays/datasets/photo2som/trainA/*')
parser.add_argument('--path_B', type=str, help='directory path of domain B', default='/home/freefridays/datasets/photo2som/trainB/*')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', default=True, action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
parser.add_argument('--resume', type=str, default='', help='snapshot path')
parser.add_argument('--tqdm', default=False, action='store_true', help='use tqdm')
parser.add_argument('--device', type=str, default='0', help='GPU ID')
parser.add_argument('--mask', default=False, action='store_true', help='use MSRA15K dataset as domain A')
parser.add_argument('--name', help='result dir name', default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str)
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
print(opt)

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import save_image
from utils import weights_init_normal
from datasets import get_dataloader

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

name = opt.name
base_dir = f'./results/{name}'
output_dir = os.path.join(base_dir, 'outputs')
snapshot_dir = os.path.join(base_dir, 'snapshots')
log_dir = os.path.join(base_dir, 'logs')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(snapshot_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# tensorboard
writer = SummaryWriter(log_dir)
logger = Logger(writer)

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batch_size, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
dataloader = get_dataloader(opt.path_A, opt.path_B, opt)

# Loss plot
# logger = Logger(opt.n_epochs, len(dataloader))

epoch = opt.epoch
iter = 0

# Resume
if opt.resume:
    state = torch.load(opt.resume)
    epoch = state['epoch']
    iter = state['iter']
    # opt.batch_size = state['batch_size']
    opt.size = state['size']
    netG_A2B.load_state_dict(state['netG_A2B'])
    netG_B2A.load_state_dict(state['netG_B2A'])
    netD_A.load_state_dict(state['netD_A'])
    netD_B.load_state_dict(state['netD_B'])
    optimizer_G.load_state_dict(state['optimizer_G'])
    optimizer_D_A.load_state_dict(state['optimizer_D_A'])
    optimizer_D_B.load_state_dict(state['optimizer_D_B'])
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step, last_epoch=epoch)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step, last_epoch=epoch)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step, last_epoch=epoch)
    fake_A_buffer = state['fake_A_buffer']
    fake_B_buffer = state['fake_B_buffer']
    print(f'Loaded from epoch: {epoch}, iter: {iter}')


# Fixed sample data
def transform_sample(sample):
    transforms_sample_ = [ transforms.ToTensor(),
                           transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]  # (0.5,0.5,0.5) (0.5,0.5,0.5, 0.5)
    for transform in transforms_sample_:
        sample = transform(sample)
    sample = torch.unsqueeze(sample, 0)
    return sample
# sample_A = transform_sample(read_image(os.path.join(opt.dataroot, 'train/A/Item_Head_G_01_Lv3_C_waifu2x_256x256_-1n_png.png'))).cuda()
# sample_B = transform_sample(read_image(os.path.join(opt.dataroot, 'train/B/Ceremonial_Dagger.png'))).cuda()
###################################

_real_A = None
_real_B = None
_fake_A = None
_fake_B = None
###### Training ######
while epoch < opt.n_epochs:
    if opt.tqdm:
        pbar = tqdm(total=len(dataloader), initial=iter)
        pbar.set_description(f'Epoch {epoch}')
    for batch in dataloader:
        if iter >= len(dataloader):
            iter = 0
            break
        # Set model input
        _real_A = real_A = Variable(input_A.copy_(batch['A']))
        _real_B = real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        _fake_B = fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
        fake_B_out = fake_B

        _fake_A = fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
        fake_A_out = fake_A

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        iter += 1

        logs = {'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
         'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A), 'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
         'loss_D': (loss_D_A + loss_D_B)}

        # print images
        if iter % 500 == 0:
          outputs = torch.cat([_real_A.detach(), _fake_B.detach(), _real_B.detach(), _fake_A.detach()], dim=0)
          save_image(outputs, f'{output_dir}/{epoch}_{iter}.png', nrow=opt.batch_size)

        # print log
        if iter % 1 == 0 and not opt.tqdm:
            print(f"[{epoch:04} epoch {iter:04} iters] G: {logs['loss_G']:.4f} | " 
            f"G_GAN: {logs['loss_G_GAN']:.4f} | G_identity: {logs['loss_G_identity']:.4f} | "
            f"G_cycle:_{logs['loss_G_cycle']:.4f} | D: {logs['loss_D']:.4f}")

        if opt.tqdm:
            pbar.update(1)
        logger.log(logs, iter)

        # Progress report (http://localhost:8097)
        # logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
        #             'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
        #             images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A_out.detach(), 'fake_B': fake_B_out.detach()})

    # save snapshot
    snapshot = {
        'epoch': epoch,
        'iter': iter,
        'batch_size': opt.batch_size,
        'size': opt.size,
        'netG_A2B': netG_A2B.state_dict(),
        'netG_B2A': netG_B2A.state_dict(),
        'netD_A': netD_A.state_dict(),
        'netD_B': netD_B.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D_A': optimizer_D_A.state_dict(),
        'optimizer_D_B': optimizer_D_B.state_dict(),
        'fake_A_buffer': fake_A_buffer,
        'fake_B_buffer': fake_B_buffer
    }
    torch.save(snapshot, os.path.join(snapshot_dir, f'{epoch}_{iter}.pt'))

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Print sample image
    # sample_fake_B = netG_A2B(sample_A).detach()
    # sample_fake_A = netG_B2A(sample_B).detach()
    # samples = {
    #     'sample_real_A': sample_A,
    #     'sample_fake_A': sample_fake_A,
    #     'sample_real_B': sample_B,
    #     'sample_fake_B': sample_fake_B
    # }
    # logger.print_samples(samples)
    epoch += 1
    iter = 0
    if opt.tqdm:
      pbar.close()
###################################
