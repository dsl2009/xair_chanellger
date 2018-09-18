from native_senet import se_resnext50_32x4d
from native_riv2 import InceptionResNetV2
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
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=(480,320)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(480, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    test_data_loader = DataLoader(test_data, batch_size=2, shuffle=True)

    model = se_resnext50_32x4d(num_classes=1000,pretrained=None)
    model.load_state_dict(torch.load('/home/dsl/all_check/se_resnext50_32x4d-a260b3a4.pth'), strict=False)
    model.last_linear = nn.Linear(2048, 61)
    model.cuda()
    state = {'learning_rate': 0.01, 'momentum': 0.9, 'decay': 0.0005}
    optimizer = torch.optim.SGD(model.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
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

    best_accuracy = 0.0
    for epoch in range(60):
        if epoch in [30]:
            state['learning_rate'] *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']
        state['epoch'] = epoch
        train()
        test()
        if state['test_accuracy'] > best_accuracy:
            best_accuracy = state['test_accuracy']
            torch.save(model.state_dict(), os.path.join('/home/dsl/all_check/aichallenger/ai_chanellger/new_newstep2', idx+'.pth'))
        with open(os.path.join('/home/dsl/all_check/aichallenger/ai_chanellger/new_newstep2', idx+'.json'),'w') as f:
            f.write(json.dumps(state))
            f.flush()
        print(state)
        print("Best accuracy: %f" % best_accuracy)
        if best_accuracy == 1.0:
            break

if __name__ == '__main__':

    train_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/base_line'
    valid_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/base_line_valid'
    run(train_dr, valid_dr, 'total', 61, str(1))