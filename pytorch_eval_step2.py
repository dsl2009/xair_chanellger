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
import json
import cv2
import shutil
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=(480,320)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(480,320)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}



labels = json.loads(open('step2_cls_id.json').read())
base_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/valid_test/'
for x in glob.glob('/home/dsl/all_check/aichallenger/ai_chanellger/step2/*.json'):
    data = json.loads(open(x).read())
    cls_name = data['cls_name']
    num_cls = data['num_class']
    model = inception_v3(num_classes=num_cls, pretrained=None, aux_logits=False)
    model.load_state_dict(torch.load(x.replace('.json','.pth')), strict=False)
    model.cuda()
    model.eval()
    print(num_cls,cls_name)
    nx = labels[cls_name]
    nd = dict()
    for l in nx:
        nd[nx[l]] = l
    print(nd)

    for k in glob.glob(os.path.join(base_dr,cls_name,'*.*')):

        img = cv2.imread(k)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(320,480))
        img = img/225.0
        img = (img-0.5)*2
        img = np.transpose(img,[2,0,1])
        img = np.expand_dims(img,0)
        ig = torch.from_numpy(img).float()

        ig = torch.autograd.Variable(ig.cuda())
        output = model(ig)
        pred = output.data.max(1)[1]
        pred_lb = pred.cpu().numpy()[0]
        new_path = os.path.join(base_dr,cls_name,nd[pred_lb])
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        shutil.move(k,new_path)





