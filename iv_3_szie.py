from native_iv3 import inception_v3
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.nn import functional as F
from progressbar import *
from matplotlib import pyplot as plt
data_transforms = {
    'train': transforms.Compose([

        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

imagenet_data = ImageFolder('D:/deep_learn_data/ai_chellenger/step2/train/番茄',
                                 transform=data_transforms['train'])
test_data = ImageFolder('D:/deep_learn_data/ai_chellenger/step2/valid/番茄',
                             transform=data_transforms['val'])
data_loader = DataLoader(imagenet_data, batch_size=1, shuffle=True)
test_data_loader = DataLoader(imagenet_data, batch_size=1, shuffle=True)
model = inception_v3(num_classes=1000, pretrained=None, aux_logits=False)

model.load_state_dict(torch.load('D:/deep_learn_data/check/inception_v3_google-1a9a5a14.pth'),strict=False)
model.fc = nn.Linear(2048, 11)
model.cuda()

def run():
    state = {'learning_rate': 0.01, 'momentum': 0.9, 'decay': 0.0005}
    optimizer = torch.optim.SGD(model.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

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
    for epoch in range(100):
        if epoch in [10, 20, 30]:
            state['learning_rate'] *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']

        state['epoch'] = epoch
        train()
        test()
        if state['test_accuracy'] > best_accuracy:
            best_accuracy = state['test_accuracy']

            torch.save(model.state_dict(), os.path.join('model_train', 'model_iv3.pytorch'))

        print(state)
        print("Best accuracy: %f" % best_accuracy)

if __name__ == '__main__':
    run()