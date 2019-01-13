'''
    Generative Adversarial Network

    version: 1.0.0
    date: Jan/13/2019
'''

import argparse
import os

import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from utils.DatasetConverter import DatasetConverter
from utils.TrainGAN import TrainGAN
from utils.discriminators.DiscriminatorDCGAN import DiscriminatorDCGAN
from utils.discriminators.DiscriminatorGAN import DiscriminatorGAN
from utils.generators.GeneratorDCGAN import GeneratorDCGAN
from utils.generators.GeneratorGAN import GeneratorGAN

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('-b', '--batch', help='batch size', type=int, default=100)
parser.add_argument('-e', '--epoch', help='number of epochs', type=int, default=20)
parser.add_argument('-d', '--dataset', help='select dataset: mnist, cifar10', type=str, default='mnist')
parser.add_argument('-gt', '--gantype', help='select gan type: gcgan, gan', type=str, default='gcgan')
parser.add_argument('-ngf', type=int, default=64)
parser.add_argument('-ndf', type=int, default=64)
args = parser.parse_args()

BATCH_SIZE = args.batch
NGF = args.ngf
NDF = args.ndf
MAX_EPOCH = args.epoch
DATA_SET = args.dataset
GAN_TYPE = args.gantype
CHANNEL_SIZE = 1
BASE_DIR = './output_images/'
SAVE_DIR = BASE_DIR + DATA_SET + '_' + GAN_TYPE

os.makedirs(SAVE_DIR, exist_ok=True)

if args.dataset == 'cifar10':
    CHANNEL_SIZE = 3

netD = DiscriminatorDCGAN(CHANNEL_SIZE, NDF)
netG = GeneratorDCGAN(100, NGF, CHANNEL_SIZE)
criterion = nn.BCELoss()

if GAN_TYPE == 'gan':
    netD = DiscriminatorGAN(64 * 64 * CHANNEL_SIZE, 1)
    netG = GeneratorGAN(100, 64 * 64 * CHANNEL_SIZE, nc=CHANNEL_SIZE)
    criterion = nn.BCELoss()
    # print(netG)
    # print(netD)

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
    for i, (data, _) in enumerate(train_loader):
        real_img = data

        errD, _, _ = train_gan.update_discriminator(real_img)
        errG, _ = train_gan.update_generator()

        if i % 100 == 0:
            print('[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f}'.format(epoch + 1, MAX_EPOCH, i + 1, len(train_loader),
                                                                        errD.item(), errG.item()))
        if epoch == 0 and i == 0:
            vutils.save_image(real_img, SAVE_DIR + '/real_sample.png', normalize=True, nrow=10)

    fake_img = train_gan.outimg()
    vutils.save_image(fake_img.detach(), SAVE_DIR + '/fake_samples_epoch_{:03d}.png'.format(epoch+1), normalize=True,
                      nrow=10)
