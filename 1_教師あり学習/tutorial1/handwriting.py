import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np

#グラフの設定
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
plt.rcParams["figure.figsize"] = [6,6]
plt.rcParams["xtick.major.pad"] = 10
plt.rcParams["figure.dpi"] = 120  # need at least 300dpi for paper
plt.rcParams["figure.autolayout"] = True

#出力の保存先の作成
folder_name = os.path.splitext(os.path.basename(__file__))[0]
folder_dir = os.path.dirname(os.path.abspath(__file__))
# dr = folder_name+"/results" #+'/'+file_name
dr = folder_dir + "/"+ folder_name + "_results"
os.makedirs(dr, exist_ok=True)

#MNISTデータの取得
BATCH_SIZE = 20
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.0,), (1.0,))])

train_set = torchvision.datasets.MNIST(root='./data', train=True,transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False,transform=transform, download=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

#ニューラルネットワークの定義
class Net(torch.nn.Module):
    def __init__(self, INPUT_FEATURES, HIDDEN, OUTPUT_FEATURES):
        super().__init__()
        self.fc1 = torch.nn.Linear(INPUT_FEATURES, HIDDEN)#linearは全結合
        self.fc2 = torch.nn.Linear(HIDDEN, OUTPUT_FEATURES)
        # self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)#隠れ層の出力：前の層から全結合した値を活性化関数ReLUに入力し計算する
        x = self.fc2(x)#出力層は全結合のみ
        # x = self.softmax(x)#出力層の活性化関数：なくてもよい
        return x

#ニューラルネットワークの各層のノード数
INPUT_FEATURES = 28 * 28
HIDDEN = 100
OUTPUT_FEATURES = 10

net = Net(INPUT_FEATURES, HIDDEN, OUTPUT_FEATURES)

criterion = torch.nn.CrossEntropyLoss()
#通常のSGD
# optimizer = optim.SGD(net.parameters(), lr=0.001)
#SGD+慣性項
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#Adam
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 10
train_loss_list=[]
test_loss_list=[]
test_accuracy_list=[]
for epoch in range(1, EPOCHS + 1):
    #学習フェーズ
    running_loss = 0.0
    for count, item in enumerate(trainloader, 1):
        inputs, labels = item
        inputs = inputs.reshape(-1, 28 * 28)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if count % 500 == 0:
            print(f'{epoch}, data: {count * BATCH_SIZE}, running_loss: {running_loss / 500:1.3f}')
            if count == 60000//BATCH_SIZE:
                train_loss_list.append(running_loss / 500.0)
            running_loss = 0.0

    #現段階での評価
    correct = 0
    total = 0
    test_loss = 0
    test_count=0

    with torch.no_grad():#勾配が計算されないモード
        for data in testloader:
            inputs, labels = data
            inputs = inputs.reshape(-1, 28 * 28)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += len(outputs)
            correct += (predicted == labels).sum().item()
            test_count+=1
    test_loss_list.append(test_loss/test_count)

    print(f'test = accuracy: {correct / total}%, loss: {test_loss/test_count:1.3f}')
    test_accuracy_list.append(correct / total)

# 結果の描画
fig = plt.figure()
plt.plot(np.arange(1, EPOCHS+1), train_loss_list, label='train_loss')
plt.plot(np.arange(1, EPOCHS+1), test_loss_list, label='test_loss')
plt.xlabel('epoch')
plt.ylabel("loss")
plt.legend()
fig.savefig(dr+"/loss.png")

fig=plt.figure()
plt.plot(np.arange(1, EPOCHS+1), test_accuracy_list)
plt.ylabel("accuracy")
plt.xlabel('epoch')
fig.savefig(dr+"/accuracy.png")
