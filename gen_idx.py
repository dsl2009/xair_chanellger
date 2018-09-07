from torchvision.models.inception import inception_v3
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.nn import functional as F
from progressbar import *
from matplotlib import pyplot as plt
import glob
import json


def step2():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size=(480, 320)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(size=(480, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }
    dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/step2/train'
    dt = dict()
    for x in os.listdir(dr):
        imagenet_data = ImageFolder(os.path.join(dr, x),
                                    transform=data_transforms['train'])
        dt[x] = imagenet_data.class_to_idx
    with open('step31_cls_id.json', 'w') as f:
        f.write(json.dumps(dt))
        f.flush()

def step3():
    dt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/org/train/AgriculturalDisease_trainingset/'
    idx = 0
    lb = dict()
    for d in os.listdir(dt):
        if len(os.listdir(os.path.join(dt,d))) ==2:
            lb[str(idx)] = d
            idx = idx+1
    with open('step3_cls_id.json', 'w') as f:
        f.write(json.dumps(lb))
        f.flush()


