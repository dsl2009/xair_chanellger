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
from torch.optim import lr_scheduler
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
data_transforms1 = {
    'train': transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

def show(data):
    ig = data.numpy()
    ig = ig[0]
    ig = np.transpose(ig, [1, 2, 0])
    ig = ig / 2 + 0.5
    plt.imshow(ig)
    plt.show()

def run(trainr,validr,name,cls_num,idx):

    imagenet_data = ImageFolder(trainr,
                                transform=data_transforms['train'])
    test_data = ImageFolder(validr,
                            transform=data_transforms['val'])
    data_loader = DataLoader(imagenet_data, batch_size=8, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    model = inception_v3(num_classes=1000, pretrained=None, aux_logits=False)
    model.load_state_dict(torch.load('/home/dsl/all_check/aichallenger/inception_v3_google-1a9a5a14.pth'), strict=False)
    model.fc = nn.Linear(2048, cls_num)
    model.cuda()
    state = {'learning_rate': 0.01, 'momentum': 0.9, 'decay': 0.0005}
    optimizer = torch.optim.SGD(model.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    state['label_ix'] = imagenet_data.class_to_idx
    state['cls_name'] = name
    def train():
        model.train()
        loss_avg = 0.0
        progress = ProgressBar()
        for (data, target) in progress(data_loader):
            data.detach().numpy()
            if data.size(0) != 8:
                break
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
            output = model(data)
            optimizer.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            loss_avg = loss_avg * 0.2 + float(loss) * 0.8
        state['train_loss'] = loss_avg
    def test():
        with torch.no_grad():
            model.eval()
            loss_avg = 0.0
            correct = 0
            for batch_idx, (data, target) in enumerate(test_data_loader):
                data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
                output = model(data)
                loss = F.cross_entropy(output, target)
                pred = output.data.max(1)[1]
                correct += float(pred.eq(target.data).sum())
                loss_avg += float(loss)
                state['test_loss'] = loss_avg / len(test_data_loader)
                state['test_accuracy'] = correct / len(test_data_loader.dataset)
            print(state['test_accuracy'])

    state['best_accuracy'] = 0.0
    for epoch in range(60):
        state['epoch'] = epoch
        train()
        test()
        scheduler.step(state['test_loss'])
        if state['test_accuracy'] > state['best_accuracy']:
            state['best_accuracy'] = state['test_accuracy']
            torch.save(model.state_dict(), os.path.join('/home/dsl/all_check/aichallenger/ai_chanellger/iv_step2', idx+'.pth'))
        with open(os.path.join('/home/dsl/all_check/aichallenger/ai_chanellger/iv_step2', idx+'.json'),'w') as f:
            f.write(json.dumps(state))
            f.flush()
        print(state)
        print("Best accuracy: %f" % state['best_accuracy'])
        if state['best_accuracy'] == 1.0:
            break

if __name__ == '__main__':
    train_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/step2_train'
    valid_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/step2_valid'
    for idx,x in enumerate(os.listdir(train_dr)):
        if True:
            rd = os.path.join(train_dr, x)
            vd = os.path.join(valid_dr, x)
            cls_nums = len(os.listdir(rd))
            run(rd, vd, x, cls_nums, str(idx))
