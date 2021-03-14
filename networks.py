import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        self.conv1  = nn.Conv2d(1,          ndf,        4, 2, 1)
        self.relu1  = nn.LeakyReLU(0.2, inplace=True)

        self.conv2  = nn.Conv2d(ndf,    ndf * 2,        4, 2, 1)
        self.bn2    = nn.BatchNorm2d(ndf * 2)
        self.relu2  = nn.LeakyReLU(0.2, inplace=True)

        self.conv3  = nn.Conv2d(ndf * 2, ndf * 4,       4, 2, 1)
        self.bn3    = nn.BatchNorm2d(ndf * 4)
        self.relu3  = nn.LeakyReLU(0.2, inplace=True)

        self.conv4  = nn.Conv2d(ndf * 4, ndf * 8,       4, 2, 1)
        self.bn4    = nn.BatchNorm2d(ndf * 8)
        self.relu4  = nn.LeakyReLU(0.2, inplace=True)

        self.conv5  = nn.Conv2d(ndf * 8,  1,            1, 1, 0)
        self.sig    = nn.Sigmoid()

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = self.conv4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.conv5(output)
        output = self.sig(output)

        return output.squeeze()

class Generator(nn.Module):
    def __init__(self, nz, ngf):
        super(Generator, self).__init__()
        self.deconv5    = nn.ConvTranspose2d(nz,        ngf * 8,    1, 1, 0)
        self.debn5      = nn.BatchNorm2d(ngf * 8)
        self.derelu5    = nn.ReLU(inplace=True)

        self.deconv4    = nn.ConvTranspose2d(ngf * 8,   ngf * 4,    4, 2, 1)
        self.debn4      = nn.BatchNorm2d(ngf * 4)
        self.derelu4    = nn.ReLU(inplace=True)

        self.deconv3    = nn.ConvTranspose2d(ngf * 4,   ngf * 2,    4, 2, 1)
        self.debn3      = nn.BatchNorm2d(ngf * 2)
        self.derelu3    = nn.ReLU(inplace=True)

        self.deconv2    = nn.ConvTranspose2d(ngf * 2,   ngf,        4, 2, 1)
        self.debn2      = nn.BatchNorm2d(ngf)
        self.derelu2    = nn.ReLU(inplace=True)

        self.deconv1    = nn.ConvTranspose2d(ngf * 1,   1,          4, 2, 1)
        self.tan1       = nn.Tanh()
  
    def forward(self, input):
        output = self.deconv5(input)
        output = self.debn5(output)
        output = self.derelu5(output)

        output = self.deconv4(output)
        output = self.debn4(output)
        output = self.derelu4(output)

        output = self.deconv3(output)
        output = self.debn3(output)
        output = self.derelu3(output)

        output = self.deconv2(output)
        output = self.debn2(output)
        output = self.derelu2(output)

        output = self.deconv1(output)
        output = self.tan1(output)

        return output

