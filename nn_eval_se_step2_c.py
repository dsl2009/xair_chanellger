from native_senet_c import se_resnext50_32x4d
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np
import glob
from torch.nn import functional as F
from progressbar import *
from matplotlib import pyplot as plt
from PIL import Image
import json
import cv2
import shutil


base_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/pred_step1'
am = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/pred_step2'
log = '/home/dsl/all_check/aichallenger/ai_chanellger/new_step3'
trans =  transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
ff = [1,2,4,5,6,7,8,9]
tt = 0
for x in ff:
    x = os.path.join('/home/dsl/all_check/aichallenger/new/se_step2/',str(x)+'.json')
    data = json.loads(open(x).read())
    cls_name = data['cls_name']
    label_ix = data['label_ix']
    print(cls_name)
    print(label_ix)
    print(len(label_ix))

    num_cls = len(label_ix)
    tt+=num_cls
    print(tt)
    model = se_resnext50_32x4d(num_classes=1000, pretrained=None)
    model.fc1 = nn.Linear(2048, 2)
    model.fc2 = nn.Linear(2, num_cls)
    model.load_state_dict(torch.load(x.replace('.json', '.pth')), strict=False)
    model.cuda()
    model.eval()

    nd = dict()
    for l in label_ix:
        nd[label_ix[l]] = l
    print(nd)

    for k in glob.glob(os.path.join(base_dr, cls_name,'*.*')):
        img = Image.open(k)
        ig = trans(img)
        ig = ig.unsqueeze(0)
        ig = torch.autograd.Variable(ig.cuda())
        _, output = model(ig)

        pred = output.data.max(1)[1]

        pred_lb = pred.cpu().numpy()[0]

        new_path = os.path.join(am, cls_name, nd[pred_lb])
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        try:
            shutil.copy(k, new_path)
        except:
            print(k)








