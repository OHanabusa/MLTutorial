# 畳み込みニューラルネットワーク(CNN)
ここでは，tutorial1と同様に手書き文字の認識を行います．ただし，ニューラルネットワーク構造として，[このサイト](https://zanote.net/python/mnist1/)を参考に畳み込みニューラルネットワークというものを使います．この畳込みという作業は画像のエッジ検出などにも使われる処理方法です．

通常のニューラルネットワークでは画像のピクセル情報をそれぞれ独立にニューラルネットワーク入力しています．そのため，１つのピクセルに対してその周辺のピクセル情報がどうなっているか考慮することができていません．そのため，３，８，９などの似ている数字が誤って認識され，精度が高くなりません．そこで，周辺のピクセル情報との関係性を考慮するために畳み込みニューラルネットワークを用います．

## プログラムの準備
`cnn_handwriting.py`をダウンロードし，実行してください．

## プログラムの説明

tutorial1との違いはニューラルネットワークの構造のみなのでそこだけ説明します．

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(1,28,3,pudding=1)
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
```
