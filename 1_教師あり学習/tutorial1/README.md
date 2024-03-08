# MNIST手書き数字を全結合型ニューラルネットワークで処理してみよう
ここでは，[このサイト](https://atmarkit.itmedia.co.jp/ait/articles/2005/21/news017.html)で紹介されていたニューラルネットワークを用いた手書き文字のクラス分類を行う．説明でわからないところがあれば，サイトを参照してみてください．

## プログラムの準備
1. まず初めにhandwriting.pyとrequirement.txtをvscodeやほかのpython環境にダウンロードしてください．
その後，ターミナルで`pip install -r requirements.txt`と入力することで必要なモジュールをダウンロードします．
この時，`pip install -r ???`の???部分はダウンロードしたrequirements.txtのパスを入力して下さい．

2. `python handwriting.py`で実行できると思います．プログラムと以下の説明を比較しながら理解してください．

## プロクラムの説明
MNISTデータベースには、下に示したような手書きの数字（と対応する正解ラベル）が訓練データとして6万個、テストデータとして1万個格納されています。この膨大な数のデータを使用して、ニューラルネットワークを用いて手書きの数字を認識してみようというのが目標です。

![image](https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/09e2a68a-fbde-4237-ac96-708b36455c59)

```python
BATCH_SIZE = 20
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.0,), (1.0,))])

train_set = torchvision.datasets.MNIST(root='./data', train=True,transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False,transform=transform, download=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
```

```python
class Net(torch.nn.Module):
    def __init__(self, INPUT_FEATURES, HIDDEN, OUTPUT_FEATURES):
        super().__init__()
        self.fc1 = torch.nn.Linear(INPUT_FEATURES, HIDDEN)
        self.fc2 = torch.nn.Linear(HIDDEN, OUTPUT_FEATURES)
        # self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x
```

一枚あたり28x28ピクセルの画像を使用するので、サイズが784(=28x28)の一次元配列に変換します。
<p align="center">
  <img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/a7633edd-fed3-4a16-8f57-ecdcc39a7abe" width="40%"><img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/9eb583f2-c84c-41a2-8339-f2e964e5588a" width="40%">
</p>
