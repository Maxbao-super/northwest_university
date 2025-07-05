import torch
import torchvision
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def firstDataSet():
    train_dataset = torchvision.datasets.ImageFolder(
        'C:\\Users\MaxBao\Desktop\\reTrainAnimals', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    torch.save(train_dataset, 'train_dataset.pth')
    return train_loader
