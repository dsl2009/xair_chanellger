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

base_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/pred/'
log = '/home/dsl/all_check/aichallenger/ai_chanellger/new_step3'
trans = transforms.Compose([
        transforms.Resize(size=(480,320)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),

    ])
for x in glob.glob('/home/dsl/all_check/aichallenger/ai_chanellger/step3_c/*.json'):
    data = json.loads(open(x).read())
    cls_name = data['cls_name']
    label_ix = data['label_ix']
    print(cls_name)
    print(label_ix)
    print(len(label_ix))

    num_cls = len(label_ix)
    model = inception_v3(num_classes=num_cls, pretrained=None, aux_logits=False)
    model.load_state_dict(torch.load(x.replace('.json', '.pth')), strict=False)
    model.cuda()
    model.eval()

    nd = dict()
    for l in label_ix:
        nd[label_ix[l]] = l
    print(nd)

    for k in glob.glob(os.path.join(base_dr,'*',cls_name,'*.*')):
        img = Image.open(k)
        ig = trans(img)
        ig = ig.unsqueeze(0)
        ig = torch.autograd.Variable(ig.cuda())
        _, output = model(ig)

        pred = output.data.max(1)[1]

        pred_lb = pred.cpu().numpy()[0]
        nn_dr = '/'.join(k.split('/')[:-1])
        new_path = os.path.join(nn_dr, nd[pred_lb])
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        try:
            shutil.move(k, new_path)
        except:
            print(k)








