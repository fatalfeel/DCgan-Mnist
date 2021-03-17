import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

def str2bool(b_str):
    if b_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser(description='CGANS MNIST')
parser.add_argument('--cuda', default=False, type=str2bool)

opt             = parser.parse_args()
device          = torch.device("cuda:0" if opt.cuda else "cpu")
nsize           = 100
batch_size      = 64
learning_rate   = 0.0002
total_epochs  	= 2000
test_samples	= 20

def NormalizeImg(img):
    nimg = (img - img.min()) / (img.max() - img.min())
    return nimg

class Discriminator(nn.Module):
	'''全连接判别器，用于1x28x28的MNIST数据,输出是数据和类别'''
	def __init__(self):
		super(Discriminator, self).__init__()

		layers = []
		# 第一层
		layers.append(nn.Linear(in_features=28*28+10, out_features=512, bias=True))
		layers.append(nn.LeakyReLU(0.2, inplace=True))
		# 第二层
		layers.append(nn.Linear(in_features=512, out_features=256, bias=True))
		layers.append(nn.LeakyReLU(0.2, inplace=True))
		# 输出层
		layers.append(nn.Linear(in_features=256, out_features=1, bias=True))
		layers.append(nn.Sigmoid())

		self.modelD = nn.Sequential(*layers)

	def forward(self, input, label):
		input 	= input.view(input.size(0), -1)
		output 	= self.modelD(torch.cat([input, label], -1))
		return output

class Generator(nn.Module):
	'''全连接生成器，用于1x28x28的MNIST数据，输入是噪声和类别'''
	def __init__(self, size):
		super(Generator, self).__init__()

		layers = []
		# 第一层
		layers.append(nn.Linear(in_features=size+10, out_features=128))
		layers.append(nn.LeakyReLU(0.2, inplace=True))
		# 第二层
		layers.append(nn.Linear(in_features=128, out_features=256))
		layers.append(nn.BatchNorm1d(256, 0.8))
		layers.append(nn.LeakyReLU(0.2, inplace=True))
		# 第三层
		layers.append(nn.Linear(in_features=256, out_features=512))
		layers.append(nn.BatchNorm1d(512, 0.8))
		layers.append(nn.LeakyReLU(0.2, inplace=True))
		# 输出层
		layers.append(nn.Linear(in_features=512, out_features=28*28))
		layers.append(nn.Tanh())

		self.modelG = nn.Sequential(*layers)

	def forward(self, input, label):
		cat_label 	= torch.cat([input, label], dim=1)
		output		= self.modelG(cat_label)
		output		= output.view(-1, 1, 28, 28)
		return output

if __name__ == '__main__':
	if not os.path.isdir('./results'):
		os.mkdir('./results')

	# 初始化构建判别器和生成器
	discriminator 	= Discriminator().to(device)
	generator 		= Generator(nsize).to(device)

	# 初始化二值交叉熵损失
	bce = torch.nn.BCELoss().to(device)
	ones = torch.ones(batch_size).to(device)
	zeros = torch.zeros(batch_size).to(device)

	# 初始化优化器，使用Adam优化器
	g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=[0.5, 0.999])
	d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=[0.5, 0.999])

	# 加载MNIST数据集
	transform   = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))])
	dataset 	= torchvision.datasets.MNIST(root='data/', train=True, transform=transform, download=True)
	dataloader 	= torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

	# 生成100个随机噪声向量
	fixed_noise 	= torch.randn([test_samples, nsize]).to(device)
	#用于生成效果图
	fixed_labels	= torch.FloatTensor(test_samples, 10).zero_()
	arrtype			= np.arange(0, 10).tolist() * (test_samples // 10)
	fixed_labels	= fixed_labels.scatter_(dim=1, index=torch.LongTensor(np.array(arrtype).reshape([test_samples, 1])), value=1)
	fixed_labels 	= fixed_labels.to(device)
	
	discriminator.train()
	generator.train()

	# 开始训练，一共训练total_epochs
	for epoch in range(total_epochs):
		# 训练一个epoch
		for i, (data_images, data_labels) in enumerate(dataloader):

			# 加载真实数据
			#data_images, data_labels = data
			data_images 	= data_images.to(device)
			# 把对应的标签转化成 one-hot 类型
			data_reshape 	= data_labels.view(-1, 1)
			real_labels 	= torch.FloatTensor(data_labels.size(0), 10).zero_()
			real_labels 	= real_labels.scatter_(dim=1, index=torch.LongTensor(data_reshape), value=1)
			real_labels 	= real_labels.to(device)

			# 生成数据
			# 用正态分布中采样batch_size个随机噪声
			noise 		= torch.randn([batch_size, nsize]).to(device)
			# 生成 batch_size 个 ont-hot 标签
			fake_labels	= torch.FloatTensor(batch_size, 10).zero_()
			fake_labels = fake_labels.scatter_(dim=1, index=torch.LongTensor(np.random.choice(10, batch_size).reshape([batch_size, 1])), value=1)
			fake_labels	= fake_labels.to(device)
			# 生成数据
			fake_images = generator(noise, fake_labels)

			# 计算判别器损失，并优化判别器
			real_loss = bce(discriminator(data_images, real_labels).squeeze(), ones)
			fake_loss = bce(discriminator(fake_images.detach(), fake_labels).squeeze(), zeros)
			d_loss = real_loss + fake_loss

			d_optimizer.zero_grad()
			d_loss.backward()
			d_optimizer.step()

			# 计算生成器损失，并优化生成器
			g_loss = bce(discriminator(fake_images, fake_labels).squeeze(), ones)

			g_optimizer.zero_grad()
			g_loss.backward()
			g_optimizer.step()

			if i%100 == 0:
				print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, total_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

		# 把生成器设置为测试模型，生成效果图并保存
		fixed_fake_images = generator(fixed_noise, fixed_labels)
		for i in range(fixed_fake_images.size(0)):
			fixed_fake_images[i,0] = 1.0 - NormalizeImg(fixed_fake_images[i,0])
		save_image(fixed_fake_images, 'results/{}.png'.format(epoch), nrow=10)
