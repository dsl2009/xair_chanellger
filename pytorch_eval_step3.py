from torchvision.models.inception import inception_v3
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
labels = json.loads(open('step3_cls_id.json').read())
base_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/valid_test/'
log = '/home/dsl/all_check/aichallenger/ai_chanellger/new_step3'
trans = transforms.Compose([
        transforms.Resize(size=(480,320)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),

    ])
for x in labels:
    if x != '0':
        continue
    cls_name = labels[x]
    model = inception_v3(num_classes=2, pretrained=None, aux_logits=False)
    model.load_state_dict(torch.load(os.path.join(log, str(x) + '.pth')), strict=True)
    model.eval()
    model.cuda()
    print(model.training)

    for k in glob.glob(os.path.join(base_dr,'*',cls_name,'*.*')):
        img = Image.open(k)
        ig = trans(img)
        ig = ig.unsqueeze(0)
        ig = torch.autograd.Variable(ig.cuda())
        output = model(ig)
        print(output)
        pred = output.data.max(1)[1]

        pred_lb = pred.cpu().numpy()[0]
        nn_dr = '/'.join(k.split('/')[:-1])
        new_path = os.path.join(nn_dr, lbs_b[pred_lb])
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        shutil.move(k, new_path)








