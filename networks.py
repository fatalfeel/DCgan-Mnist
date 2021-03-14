import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        '''self.network = nn.Sequential(nn.Conv2d(nc, ndf,         4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Conv2d(ndf, ndf * 2,     4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ndf * 2),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Conv2d(ndf*2, ndf*4,     3, 2, 1, bias=False),
                                    nn.BatchNorm2d(ndf * 4),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Conv2d(ndf * 4, 1,       4, 1, 0, bias=False),
                                    nn.Sigmoid())'''

        self.conv1  = nn.Conv2d(nc, ndf,            4, 2, 1)
        self.relu1  = nn.LeakyReLU(0.2)

        self.conv2  = nn.Conv2d(ndf, ndf * 2,       4, 2, 1)
        self.bn2    = nn.BatchNorm2d(ndf * 2)
        self.relu2  = nn.LeakyReLU(0.2)

        self.conv3  = nn.Conv2d(ndf * 2, ndf * 4,   3, 2, 1)
        self.bn3    = nn.BatchNorm2d(ndf * 4)
        self.relu3  = nn.LeakyReLU(0.2)

        self.conv4  = nn.Conv2d(ndf * 4, 1,   4, 1, 0)
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
        output = self.sig(output)

        return output.view(-1, 1).squeeze()

class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(Generator, self).__init__()
        '''self.network = nn.Sequential(nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0, bias=False),
                                        nn.BatchNorm2d(ngf*4),
                                        nn.ReLU(True),
    
                                        nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
                                        nn.BatchNorm2d(ngf*2),
                                          nn.ReLU(True),
    
                                        nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
                                          nn.BatchNorm2d(ngf),
                                       nn.ReLU(True),

                                          nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                                          nn.Tanh())'''

        self.deconv4    = nn.ConvTranspose2d(nz, ngf * 4,  4, 1, 0)
        self.debn4      = nn.BatchNorm2d(ngf * 4)
        self.derelu4    = nn.ReLU()

        self.deconv3    = nn.ConvTranspose2d(ngf * 4, ngf * 2,  3, 2, 1)
        self.debn3      = nn.BatchNorm2d(ngf * 2)
        self.derelu3    = nn.ReLU()

        self.deconv2    = nn.ConvTranspose2d(ngf * 2, ngf,      4, 2, 1)
        self.debn2      = nn.BatchNorm2d(ngf)
        self.derelu2    = nn.ReLU()

        self.deconv1    = nn.ConvTranspose2d(ngf, nc,           4, 2, 1)
        self.tan1       = nn.Tanh()
  
    def forward(self, input):
      output = self.deconv4(input)
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

