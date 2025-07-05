import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from torchvision import transforms

device = torch.device("cpu")

batch_size = 32

# Compose定义了一系列transform，此操作相当于将多个transform一并执行
transform = transforms.Compose([
    transforms.ToTensor(),
    # mnist是灰度图，此处只将一个通道标准化
    transforms.Normalize(mean=0.5,
                         std=0.5)
])

# 设定数据集 第一次使用时`download=True`进行MNIST的数据集下载,若根目录有则设置为False
mnist_data = torchvision.datasets.MNIST("./mnist_data", train=True, download=False, transform=transform)

# 加载数据集，按照上述要求，shuffle本意为洗牌，这里指打乱顺序，很形象
dataloader = torch.utils.data.DataLoader(dataset=mnist_data,
                                         batch_size=batch_size,
                                         shuffle=True)
# MNIST的数据集是28*28的
image_size = 784
hidden_size = 256

# Discriminator 判别器
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()  # sigmoid结果为（0，1）
)

# Generator 生成器
latent_size = 64  # latent_size，相当于初始噪声的维数
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()  # 转换至（-1，1）
)

D.load_state_dict(torch.load('discriminator.pth'))
G.load_state_dict(torch.load('generator.pth'))
# 向G输入一个噪声，观察生成的图片

for i in range(5):
    z = torch.randn(1, latent_size).to(device)
    fake_images = G(z).view(28, 28).data.cpu().numpy()
    plt.imshow(fake_images, cmap=plt.cm.gray)
    plt.show()
    plt.imshow(next(iter(dataloader))[0][0][0], cmap=plt.cm.gray)
    plt.show()
