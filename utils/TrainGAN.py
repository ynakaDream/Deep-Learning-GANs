import torch


class TrainGAN:
    def __init__(self, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('DEVICE:', self.device)
        self.batch_size = batch_size

        self.noise = None
        self.real_target = None
        self.fake_target = None
        self.fake_image = None

        self.optimizerD = None
        self.optimizerG = None
        self.criterion = None

        self.netD = None
        self.netG = None

        self.fixed_noise = torch.randn(self.batch_size, 100, 1, 1, device=self.device)

        self.target_setting()

    def target_setting(self):
        '''
        Define real(=1) and fake(=0) target tensor
        :return: None
        '''
        self.real_target = torch.full((self.batch_size,), 1., device=self.device)
        self.fake_target = torch.full((self.batch_size,), 0., device=self.device)

    def networks(self, netD, netG):
        '''
        Define neural networks for discriminators and generator
        :param netD: neural network for discriminators
        :param netG: neural network for generator
        :return: None
        '''
        self.netD = netD.to(self.device)
        self.netG = netG.to(self.device)

    def optimizers(self, optD, optG):
        '''
        Define optimizers for discriminators and generator
        :param optD: optimizer for discriminators
        :param optG: optimizer for generator
        :return: None
        '''
        self.optimizerD = optD
        self.optimizerG = optG

    def loss(self, criterion):
        '''
        Define loss function
        :param criterion: loss function
        :return: None
        '''
        self.criterion = criterion

    def update_discriminator(self, real_img):
        '''
        Update Discriminator
        :param real_img:
        :return: errD, D_x, D_G_z1
        '''
        real_img = real_img.to(self.device)

        noise = torch.randn(self.batch_size, 100, 1, 1, device=self.device)

        self.netD.zero_grad()
        output = self.netD(real_img)
        errD_real = self.criterion(output, self.real_target)
        errD_real.backward()

        self.fake_img = self.netG(noise)
        output = self.netD(self.fake_img.detach())
        errD_fake = self.criterion(output, self.fake_target)
        errD_fake.backward()

        errD = errD_real + errD_fake
        D_x = output.mean().item()
        D_G_z1 = output.mean().item()

        self.optimizerD.step()

        return errD, D_x, D_G_z1

    def update_generator(self):
        '''
        Update Generator
        :return: errG, D_G_z2
        '''
        self.netG.zero_grad()
        output = self.netD(self.fake_img)
        errG = self.criterion(output, self.real_target)
        errG.backward()
        D_G_z2 = output.mean().item()

        self.optimizerG.step()

        return errG, D_G_z2

    def output_images(self):
        fake_image = self.netG(self.fixed_noise)
        return fake_image
