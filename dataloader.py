import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def Construct_DataLoader(dataset, batchsize):
    return DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True)

# transform = transforms.Compose([
#     transforms.Resize(96),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# def LoadCIFAR10(download=False):
#     # Load CIFAR-10 dataset
#     train_dataset = torchvision.datasets.CIFAR10(root='../Data/CIFAR10', train=True, transform=transform, download=download)
#     test_dataset = torchvision.datasets.CIFAR10(root='../Data/CIFAR10', train=False, transform=transform)
#     return train_dataset, test_dataset

def LoadCIFAR10(download=False):
    train_dataset = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
