import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import datetime

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['mathtext.default'] = "it"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True  # x軸補助目盛りの追加
plt.rcParams["ytick.minor.visible"] = True  # y軸補助目盛りの追加
plt.rcParams["xtick.major.width"] = 1.0  # x軸主目盛り線の線幅
plt.rcParams["ytick.major.width"] = 1.0  # y軸主目盛り線の線幅
plt.rcParams["xtick.minor.width"] = 1.0  # x軸補助目盛り線の線幅
plt.rcParams["ytick.minor.width"] = 1.0  # y軸補助目盛り線の線幅
plt.rcParams["xtick.major.size"] = 10  # x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 10  # y軸主目盛り線の長さ
plt.rcParams["xtick.minor.size"] = 5  # x軸補助目盛り線の長さ
plt.rcParams["ytick.minor.size"] = 5
plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = [6,4]
plt.rcParams["xtick.major.pad"] = 10
plt.rcParams["figure.dpi"] = 120  # need at least 300dpi for paper
plt.rcParams["figure.autolayout"] = True

# file_name = datetime.datetime.now().strftime('%Y_%m_%d'+'/%m_%d_%H_%M_%S')
folder_name = os.path.splitext(os.path.basename(__file__))[0] #実行しているファイルの名前
folder_dir = os.path.dirname(os.path.abspath(__file__)) #実行しているファイルのパス（どこにあるか）
# dr = folder_name+"/results" #+'/'+file_name
dr = folder_dir + "/"+ folder_name + "_results"
os.makedirs(dr, exist_ok=True)

# train用データ
batch_size = 200
train_loader = DataLoader(
    datasets.MNIST('./',
                    train=True,
                    download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ])),
    batch_size=batch_size,
    shuffle=True
    )

# eval用データ
test_loader = DataLoader(
    datasets.MNIST('./',
                    train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ])),
    batch_size=batch_size,
    shuffle=True)

data_loader_dict = {'train': train_loader, 'test': test_loader}

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(1,28,3,padding=1)
        self.conv2 = nn.Conv2d(28,32,3)

        self.fc1 = nn.Linear(32*6*6, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.pool(x)
        x = x.reshape(x.size()[0], -1)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)
    


train_loss_list = []
test_loss_list = []
test_acc_list = []

net = Net()

optimizer = torch.optim.AdamW(params=net.parameters(), lr=0.005) # 最適化アルゴリズムを選択

epochs = 10
for epoch in range(epochs):
    """ Training """
    loss = None
    net.train()
    for i, (data, target) in enumerate(data_loader_dict['train']):
        optimizer.zero_grad()
        output = net(data)
        loss = f.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        print("\rTraining log: {0} epoch ({1} / {2}). Loss: {3}%".format(epoch+1, (i+1)*batch_size, len(train_loader)*batch_size, loss.item()), end="")

    train_loss_list.append(loss.detach().numpy())

    """ eval """
    net.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader_dict['test']:
            output = net(data)
            test_loss += f.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= 10000

    print("")
    print('Test loss (avg): {0}, Accuracy: {1}'.format(test_loss, correct / 10000), sep="")

    test_loss_list.append(test_loss)
    test_acc_list.append(correct / 10000)
    
# 結果の出力と描画
# print("test_acc", test_acc_list)
fig = plt.figure()
plt.plot(range(1, epochs+1), train_loss_list, label='train_loss')
plt.plot(range(1, epochs+1), test_loss_list, label='test_loss')
plt.xlabel('epoch')
plt.legend()
fig.savefig(dr+"/loss.png")

fig=plt.figure()
plt.plot(range(1, epochs+1), test_acc_list)
# plt.title('test accuracy')
plt.xlabel('epoch')
fig.savefig(dr+"/accuracy.png")
