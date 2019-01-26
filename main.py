'''
    Generative Adversarial Network

    version: 1.1.1

    date: Jan/13/2019    version: 1.0.0
          Jan/14/2019             1.1.0 deal with STL10 dataset
          Jan/16/2019             1.1.1 bug fix
'''

import argparse
import sys
import os
from tqdm import tqdm

import torch.optim as optim
import torchvision.utils as vutils

from utils.DatasetConverter import DatasetConverter
from utils.TrainGAN import TrainGAN
from utils.Discriminator import *
from utils.Generator import *

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('-b', '--batch', help='batch size', type=int, default=100)
parser.add_argument('-e', '--epoch', help='number of epochs', type=int, default=20)
parser.add_argument('-d', '--dataset', help='select dataset: mnist, cifar10, stl10', type=str, default='mnist')
parser.add_argument('-gt', '--gantype', help='select gan type: dcgan, gan, dcgan_pixel', type=str, default='dcgan_pixel')
parser.add_argument('-ngf', type=int, default=64)
parser.add_argument('-ndf', type=int, default=64)
args = parser.parse_args()

BATCH_SIZE = args.batch
NGF = args.ngf
NDF = args.ndf
MAX_EPOCH = args.epoch
DATA_SET = args.dataset
GAN_TYPE = args.gantype
CHANNEL_SIZE = None
BASE_DIR = './output_images/'
SAVE_DIR = BASE_DIR + DATA_SET + '_' + GAN_TYPE

os.makedirs(SAVE_DIR, exist_ok=True)

netD, netG = None, None
criterion = None

if DATA_SET == 'mnist':
    CHANNEL_SIZE = 1
elif DATA_SET == 'cifar10' or DATA_SET == 'stl10':
    CHANNEL_SIZE = 3
else:
    print('Dataset {}: Not Found:'.format(DATA_SET))
    sys.exit(1)

if GAN_TYPE == 'dcgan':
    netD = DiscriminatorDCGAN(CHANNEL_SIZE, NDF)
    netG = GeneratorDCGAN(100, NGF, CHANNEL_SIZE)
    criterion = nn.BCELoss()
elif GAN_TYPE == 'dcgan_pixel':
    netD = DiscriminatorDCGAN(CHANNEL_SIZE, NDF)
    netG = GeneratorDCGANPixel(100, NGF, CHANNEL_SIZE)
    criterion = nn.MSELoss()
elif GAN_TYPE == 'gan':
    netD = DiscriminatorGAN(64 * 64 * CHANNEL_SIZE, 1)
    netG = GeneratorGAN(100, 64 * 64 * CHANNEL_SIZE, nc=CHANNEL_SIZE)
    criterion = nn.BCELoss()
else:
    print('Gan type {}: Not Found'.format(GAN_TYPE))
    sys.exit(1)

print('BATCH SIZE:', BATCH_SIZE)
print('EPOCHS:', MAX_EPOCH)
print('DATASET:', DATA_SET.upper())
print('GAN type:', GAN_TYPE.upper())

optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

train_gan = TrainGAN(BATCH_SIZE)
train_gan.networks(netD, netG)
train_gan.optimizers(optimizerD, optimizerG)
train_gan.loss(criterion)

(train_loader, _) = DatasetConverter(args.dataset, batch_size=BATCH_SIZE).run()

for epoch in range(MAX_EPOCH):
    with tqdm(total=len(train_loader), leave=False, unit_scale=True) as qbar:
        for i, (data, _) in enumerate(train_loader):
            real_img = data

            errD, _, _ = train_gan.update_discriminator(real_img)
            errG, _ = train_gan.update_generator()

            qbar.set_description('Epoch {}/{}'.format(epoch+1, MAX_EPOCH))

            if epoch == 0 and i == 0:
                vutils.save_image(real_img, SAVE_DIR + '/real_sample.png', normalize=True, nrow=10)

            qbar.update(1)

    print('Epoch {}/{} Loss_D: {:.3f} Loss_G: {:.3f}'.format(epoch+1, MAX_EPOCH, errD.item(), errG.item()))

    fake_img = train_gan.output_images()
    vutils.save_image(fake_img.detach(), SAVE_DIR + '/fake_samples_epoch_{:03d}.png'.format(epoch+1), normalize=True,
                      nrow=10)
