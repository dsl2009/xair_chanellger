import glob
import cv2
from skimage import io
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=(480,320)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(480,320)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}
imagenet_data = ImageFolder('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/valid_test/番茄',
                                 transform=data_transforms['val'])

data_loader = DataLoader(imagenet_data, batch_size=8, shuffle=True)
for x , l in data_loader:
    print(x,l)
