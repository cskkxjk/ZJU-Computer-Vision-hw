# python 3
# utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 16
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001
momentum = 0.92

transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
image_datasets = {x: torchvision.datasets.CIFAR10(root='./data', train=(x == 'train'),
                                                  download=False, transform=transforms) for x in ['train', 'test']}
dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=(x == 'train')) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes


class Fire(nn.Module):

    def __init__(self, inchn, sqzout_chn, exp1x1out_chn, exp3x3out_chn):
        super(Fire, self).__init__()
        self.inchn = inchn
        self.squeeze = nn.Conv2d(inchn, sqzout_chn, kernel_size=1)
        self.squeeze_act = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(sqzout_chn, exp1x1out_chn, kernel_size=1)
        self.expand1x1_act = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(sqzout_chn, exp3x3out_chn, kernel_size=3, padding=1)
        self.expand3x3_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_act(self.squeeze(x))
        return torch.cat([
            self.expand1x1_act(self.expand1x1(x)),
            self.expand3x3_act(self.expand3x3(x)),
        ], 1)


class Sqznet(nn.Module):

    # CIFAR10 need 10 classes
    def __init__(self, num_class=10):
        super(Sqznet, self).__init__()
        self.num_class = num_class
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        final_conv = nn.Conv2d(512, self.num_class, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_class)


def imshow(img, title=None):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.show()


def train_model(model, criterion, optimizer, scheduler, device, num_epochs=1):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train(True)   # train mode
            else:
                model.train(False)  # test mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            iter = 0
            for data in dataloader[phase]:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                preds = torch.max(outputs.data, 1)[1]
                loss = criterion(outputs, labels)
                print("phase:%s, epoch:%d/%d Iter %d: loss=%s" % (
                    phase, epoch, num_epochs - 1, iter, str(loss.data.cpu().numpy())))
                # backward + optimizer only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data.cpu()
                running_corrects += sum(preds == labels.data)
                iter += 1
            scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print('-' * 10)
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    # show a batch of image
    images, classes = next(iter(dataloader['train']))
    out = torchvision.utils.make_grid(images)
    imshow(out, title=[class_names[x] for x in classes])

    # look at the sqznet model
    model = Sqznet()
    # print(model)

    # train
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, DEVICE, EPOCHS)

    # model save
    torch.save(model, 'Sqznet.pth')
