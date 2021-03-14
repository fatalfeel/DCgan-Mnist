import torch
import matplotlib.pyplot as plt
import math
import itertools
import imageio
import natsort
from glob import glob
from torchvision import datasets, transforms

kwargs = {}

def get_data_loader(batch_size):
    # MNIST Dataset
    transform       = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307, ), std=(0.3081,))])
    train_dataset   = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
    train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def NormalizeImg(img):
    nimg = (img - img.min()) / (img.max() - img.min())
    return nimg

def generate_images(epoch, path, fixed_noise, num_test_samples, nsize, netG, device, use_fixed=False):
    title               = ''
    local_noise         = torch.randn(num_test_samples, nsize, 1, 1, device=device)
    size_figure_grid    = int(math.sqrt(num_test_samples))

    if use_fixed:
        generated_fake_images = netG(fixed_noise)
        path += 'fixed_noise/'
        title = 'Fixed Noise'
    else:
        generated_fake_images = netG(local_noise)
        path += 'variable_noise/'
        title = 'Variable Noise'
  
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid)
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)

    for k in range(num_test_samples):
        i = k//4
        j = k%4
        ax[i,j].cla()
        nimg = 1.0 - NormalizeImg(generated_fake_images[k, 0].detach().cpu()) #reverse black white
        ax[i, j].imshow(nimg.numpy(), cmap='gray')

    label = 'Epoch_{}'.format(epoch+1)
    fig.text(0.5, 0.04, label, ha='center')
    fig.suptitle(title)
    fig.savefig(path+label+'.png')

def save_gif(path, fps, fixed_noise=False):
    if fixed_noise==True:
        path += 'fixed_noise/'
    else:
        path += 'variable_noise/'
    images = glob(path + '*.png')
    images = natsort.natsorted(images)
    gif = []

    for image in images:
        gif.append(imageio.imread(image))
    imageio.mimsave(path+'animated.gif', gif, fps=fps)

    

    