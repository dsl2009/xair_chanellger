from native_iv3_center import inception_v3
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
from losses.center_loss import CenterLoss

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(299),
        transforms.RandomCrop(299),
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
def show(data):
    ig = data.numpy()
    ig = ig[0]
    ig = np.transpose(ig, [1, 2, 0])
    ig = ig / 2 + 0.5
    plt.imshow(ig)
    plt.show()
def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00']
    plt.clf()
    for i in range(2):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1'], loc = 'upper right')
    plt.xlim(xmin=-5,xmax=5)
    plt.ylim(ymin=-5,ymax=5)
    plt.text(-4.8,4.6,"epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)

def run(trainr,validr,names,cls_name):

    imagenet_data = ImageFolder(trainr,
                                transform=data_transforms['train'])
    test_data = ImageFolder(validr,
                            transform=data_transforms['val'])
    data_loader = DataLoader(imagenet_data, batch_size=8, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=8, shuffle=True)
    model = inception_v3(num_classes=1000, pretrained=None, aux_logits=False)
    model.load_state_dict(torch.load('/home/dsl/all_check/aichallenger/inception_v3_google-1a9a5a14.pth'), strict=False)
    model.fc1 = nn.Linear(2048, 2)
    model.fc2 = nn.Linear(2, 2)
    model.cuda()
    state = {'learning_rate': 0.01, 'momentum': 0.9, 'decay': 0.0005}
    optimizer = torch.optim.SGD(model.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)



    state['label_ix'] = imagenet_data.class_to_idx
    state['cls_name'] = cls_name
    centerloss = CenterLoss(2,2)
    centerloss.cuda()
    optimzer_center = torch.optim.SGD(centerloss.parameters(), lr=0.5)

    def train():
        model.train()
        loss_avg = 0.0
        progress = ProgressBar()
        ip1_loader = []
        idx_loader = []
        for (data, target) in progress(data_loader):
            data.detach().numpy()
            if data.size(0) != 8:
                break
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
            fc1, output = model(data)
            optimizer.zero_grad()
            optimzer_center.zero_grad()

            loss = F.cross_entropy(output, target) + centerloss(target, fc1)*0.5

            loss.backward()
            optimizer.step()
            optimzer_center.step()

            ip1_loader.append(fc1)
            idx_loader.append((target))

            loss_avg = loss_avg * 0.2 + float(loss) * 0.8
        feat = torch.cat(ip1_loader, 0)
        labels = torch.cat(idx_loader, 0)

        visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
        state['train_loss'] = loss_avg
    def test():
        with torch.no_grad():
            model.eval()
            loss_avg = 0.0
            correct = 0
            for batch_idx, (data, target) in enumerate(test_data_loader):
                data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
                fc1, output = model(data)
                loss = F.cross_entropy(output, target)
                pred = output.data.max(1)[1]
                correct += float(pred.eq(target.data).sum())
                loss_avg += float(loss)
                state['test_loss'] = loss_avg / len(test_data_loader)
                state['test_accuracy'] = correct / len(test_data_loader.dataset)
            print(state['test_accuracy'])

    state['best_accuracy'] = 0.0
    for epoch in range(30):
        if epoch in [10,20,30]:
            state['learning_rate'] *= 0.34
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']
        state['epoch'] = epoch
        train()
        test()
        if state['test_accuracy'] >= state['best_accuracy']:
            state['best_accuracy'] = state['test_accuracy']
            torch.save(model.state_dict(), os.path.join('/home/dsl/all_check/aichallenger/ai_chanellger/step3_c', names+'.pth'))
        with open(os.path.join('/home/dsl/all_check/aichallenger/ai_chanellger/step3_c', names+'.json'),'w') as f:
            f.write(json.dumps(state))
            f.flush()
        print(state)
        print("Best accuracy: %f" % state['best_accuracy'])
        if state['best_accuracy'] == 1.0 and epoch>8:
            break

if __name__ == '__main__':
    train_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/step3/'
    valid_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/step3_valid/'
    for idx, x in enumerate(os.listdir(train_dr)):
        fu_dr = os.path.join(train_dr,x)
        fu_valid_dr = os.path.join(valid_dr,x)
        if idx>0:
            for ii, k in enumerate(os.listdir(fu_dr)):
                if True:
                    print(x)
                    rd = os.path.join(fu_dr, k)
                    vd = os.path.join(fu_valid_dr, k)
                    cls_nums = len(os.listdir(rd))
                    run(rd, vd, str(idx)+str(ii), k)
