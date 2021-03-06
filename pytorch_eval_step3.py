from native_iv3_center import inception_v3
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
lbs_b = {0:'一般',  1:'严重'}
base_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/pred_step1'
am = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/pred_step2'

log = '/home/dsl/all_check/aichallenger/ai_chanellger/new_step3'
trans = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),

    ])

l1 = ['/home/dsl/all_check/aichallenger/new/step2/樱桃.json', '/home/dsl/all_check/aichallenger/new/step2/桃子.json']
#l1 = glob.glob('/home/dsl/all_check/aichallenger/new/step2/*.json')

for x in l1:
    data = json.loads(open(x).read())
    cls_name = x.split('/')[-1].split('.')[0]
    label_ix = data['label_ix']
    print(cls_name)
    print(label_ix)
    print(len(label_ix))

    num_cls = len(label_ix)

    model = inception_v3(num_classes=1000, pretrained=None, aux_logits=False)
    model.fc1 = nn.Linear(2048, 2)
    model.fc2 = nn.Linear(2, num_cls)
    model.load_state_dict(torch.load(x.replace('.json', '.pth')), strict=True)
    model.cuda()
    model.eval()

    nd = dict()
    for l in label_ix:
        nd[label_ix[l]] = l
    print(nd)

    for k in glob.glob(os.path.join(base_dr,cls_name,'*.*')):
        img = Image.open(k)
        ig = trans(img)
        ig = ig.unsqueeze(0)
        ig = torch.autograd.Variable(ig.cuda())
        _, output = model(ig)

        pred = output.data.max(1)[1]

        pred_lb = pred.cpu().numpy()[0]

        new_path = os.path.join(am, cls_name,nd[pred_lb])
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        try:
            shutil.copy(k, new_path)
        except:
            print(k)








