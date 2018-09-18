from native_iv3 import inception_v3
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.nn import functional as F
from progressbar import *
from matplotlib import pyplot as plt
import numpy as np
import json
import h5py
from PIL import Image
import glob
f = h5py.File('taozi.h5',mode='w')
trans = transforms.Compose([
        transforms.Resize(size=(480,320)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])




model = inception_v3(num_classes=1000, pretrained=None, aux_logits=False)
model.load_state_dict(torch.load('/home/dsl/all_check/aichallenger/inception_v3_google-1a9a5a14.pth'), strict=False)
model.eval()
data = []
lbs = []

for p in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/step3/樱桃/樱桃白粉病/*/*.*'):
    ig = Image.open(p)
    igs = trans(ig)
    igs = torch.unsqueeze(igs,0)
    igs.cuda()
    _ ,out = model(igs)
    out = torch.squeeze(out)
    data.append(out.cpu().detach().numpy())
    if '一般' in p:
        lbs.append(0)
    else:
        lbs.append(1)

f['data'] = np.asarray(data)
f['label'] = np.asarray(lbs)
f.flush()
f.close()