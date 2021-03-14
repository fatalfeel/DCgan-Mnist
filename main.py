import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from networks import Generator, Discriminator
from utils import get_data_loader, generate_images, save_gif

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser(description='DCGANS MNIST')
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--ndf', type=int, default=16, help='Number of features to be used in Discriminator network')
parser.add_argument('--ngf', type=int, default=16, help='Number of features to be used in Generator network')
parser.add_argument('--nsize', type=int, default=100, help='Size of the noise')
parser.add_argument('--d-lr', type=float, default=0.0002, help='Learning rate for the discriminator')
parser.add_argument('--g-lr', type=float, default=0.0002, help='Learning rate for the generator')
parser.add_argument('--nc', type=int, default=1, help='Number of input channels. Ex: for grayscale images: 1 and RGB images: 3 ')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--num-test-samples', type=int, default=16, help='Number of samples to visualize')
parser.add_argument('--output-path', type=str, default='./results/', help='Path to save the images')
parser.add_argument('--fps', type=int, default=5, help='frames-per-second value for the gif')
parser.add_argument('--use-fixed', default=True, type=str2bool, help='Boolean to use fixed noise or not')
parser.add_argument('--cuda', default=False, type=str2bool)

opt     = parser.parse_args()
device  = torch.device("cuda:0" if opt.cuda else "cpu")

if __name__ == '__main__':

    # Gather MNIST Dataset    
    train_loader = get_data_loader(opt.batch_size)

    # Define Discriminator and Generator architectures
    netG = Generator(opt.nc, opt.nsize, opt.ngf).to(device)
    netD = Discriminator(opt.nc, opt.ndf).to(device)

    # loss function
    lossfuncD = nn.BCELoss()
    lossfuncG = nn.BCELoss()

    # optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=opt.d_lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.g_lr)

    netD.train()
    netG.train()
    
    # initialize other variables
    num_batches = len(train_loader)
    fixed_noise = torch.randn(opt.num_test_samples, 100, 1, 1, device=device)

    real_label = torch.ones(opt.batch_size, dtype=torch.float32).to(device)
    fake_label = torch.zeros(opt.batch_size, dtype=torch.float32).to(device)

    for epoch in range(opt.num_epochs):
        for i, (real_images, mnist_labels) in enumerate(train_loader):
            ##############################
            #   Training discriminator   #
            ##############################
            real_images     = real_images.to(device)
            noise           = torch.randn(opt.batch_size, opt.nsize, 1, 1).to(device)
            noise_images    = netG(noise)

            lossD_real      = lossfuncD(netD(real_images), real_label)
            lossD_fake      = lossfuncD(netD(noise_images.detach()), fake_label)

            lossD = lossD_real + lossD_fake

            netD.zero_grad()
            lossD.backward()
            optimizerD.step()

            ##########################
            #   Training generator   #
            ##########################
            lossG  = lossfuncG(netD(noise_images), real_label)
            netG.zero_grad()
            lossG.backward()
            optimizerG.step()

            if i%100 == 0:
                print('Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}'.format(epoch+1,
                                                                                           opt.num_epochs,
                                                                                           i+1,
                                                                                           num_batches,
                                                                                           lossD.item(),
                                                                                           lossG.item()))

        generate_images(epoch, opt.output_path, fixed_noise, opt.num_test_samples, netG, device, use_fixed=opt.use_fixed)

    # Save gif:
    save_gif(opt.output_path, opt.fps, fixed_noise=opt.use_fixed)