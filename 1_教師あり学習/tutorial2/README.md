# 畳み込みニューラルネットワーク(CNN)
ここでは，tutorial1と同様に手書き文字の認識を行います．ただし，ニューラルネットワーク構造として，[このサイト](https://zanote.net/python/mnist1/)を参考に畳み込みニューラルネットワークというものを使います．この畳込みという作業は画像のエッジ検出などにも使われる処理方法です．

通常のニューラルネットワークでは画像のピクセル情報をそれぞれ独立にニューラルネットワーク入力しています．そのため，１つのピクセルに対してその周辺のピクセル情報がどうなっているか考慮することができていません．そのため，３，８，９などの似ている数字が誤って認識され，精度が高くなりません．そこで，周辺のピクセル情報との関係性を考慮するために畳み込みニューラルネットワークを用います．

## プログラムの準備
`cnn_handwriting.py`をダウンロードし，実行してください．

## プログラムの説明

tutorial1との違いはニューラルネットワークの構造と損失関数のnlllossのみなので，そこだけ説明します．

ニューラルネットワークの構造としては，２層の畳み込みニューラルネットワークと２層の全結合ニューラルネットワークで構成されている．

```python
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
```

nll_lossとは，参考資料にあるように，出力層のlog_softmaxを用いると交差エントロピー誤差と同様の損失関数となります．

## 学習結果
学習の過程とそのプロットを以下に示す．プロットはAccuracyを左図，lossを右図とする．これらより，文字認識の精度は最大で99.1％程度となったことが分かる．一方，Accuracy，lossの値はどちらも不安定な変化となっていることが分かる．
```python
Training log: 1 epoch (60000 / 60000). Loss: 0.037224993109703064%
Test loss (avg): 0.04151938211917877, Accuracy: 0.9863
Training log: 2 epoch (60000 / 60000). Loss: 0.05282442271709442%%%
Test loss (avg): 0.03715595973134041, Accuracy: 0.9867
Training log: 3 epoch (60000 / 60000). Loss: 0.013176638633012772%%
Test loss (avg): 0.03639041123986244, Accuracy: 0.9871
Training log: 4 epoch (60000 / 60000). Loss: 0.022388048470020294%%
Test loss (avg): 0.03184212540388107, Accuracy: 0.9888
Training log: 5 epoch (60000 / 60000). Loss: 0.014183281920850277%%
Test loss (avg): 0.04370289291143417, Accuracy: 0.9861
Training log: 6 epoch (60000 / 60000). Loss: 0.001061366405338049%%
Test loss (avg): 0.030838098107278347, Accuracy: 0.9907
Training log: 7 epoch (60000 / 60000). Loss: 0.011724426411092281%%%
Test loss (avg): 0.04036673328280449, Accuracy: 0.9888
Training log: 8 epoch (60000 / 60000). Loss: 0.020657695829868317%%%
Test loss (avg): 0.055643657529354094, Accuracy: 0.9858
Training log: 9 epoch (60000 / 60000). Loss: 0.011351628229022026%%%
Test loss (avg): 0.03828562349230051, Accuracy: 0.9912
Training log: 10 epoch (60000 / 60000). Loss: 0.000987999141216278%%%
Test loss (avg): 0.040668908001482484, Accuracy: 0.9902
```

</p>
<p align="center">
  <img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/188c3906-ed8b-461f-a546-2a4090caefc0" width="50%"><img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/f7646162-b6de-4acc-85f0-77b95db6ae9a" width="50%">
</p>

次に，scheduler1とscheduler1.step()を用いることにとって，毎epochの最後に学習率を0.5倍にして学習を行った．下図より，学習率を一定にした場合と比べて安定な学習が行えたことが分かる．

```python
Training log: 1 epoch (60000 / 60000). Loss: 0.05678172409534454%%
Test loss (avg): 0.05794725885391235, Accuracy: 0.9812
Training log: 2 epoch (60000 / 60000). Loss: 0.02136416919529438%%%
Test loss (avg): 0.034433253765106205, Accuracy: 0.9874
Training log: 3 epoch (60000 / 60000). Loss: 0.006577419117093086%%
Test loss (avg): 0.029183958357572554, Accuracy: 0.9898
Training log: 4 epoch (60000 / 60000). Loss: 0.004205705597996712%%
Test loss (avg): 0.02749570277929306, Accuracy: 0.991
Training log: 5 epoch (60000 / 60000). Loss: 0.0014781500212848186%
Test loss (avg): 0.02610298503637314, Accuracy: 0.991
Training log: 6 epoch (60000 / 60000). Loss: 0.003578977659344673%%%
Test loss (avg): 0.02527144007384777, Accuracy: 0.9918
Training log: 7 epoch (60000 / 60000). Loss: 0.0023889292497187853%
Test loss (avg): 0.025273167470097543, Accuracy: 0.9917
Training log: 8 epoch (60000 / 60000). Loss: 0.002778843278065324%%%
Test loss (avg): 0.025554137930274008, Accuracy: 0.9914
Training log: 9 epoch (60000 / 60000). Loss: 0.0032909789588302374%%
Test loss (avg): 0.025495155468210576, Accuracy: 0.992
Training log: 10 epoch (60000 / 60000). Loss: 0.0006302946130745113%%
Test loss (avg): 0.02545464370548725, Accuracy: 0.992
```
<p align="center">
  <img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/4ed40275-fc65-4bc6-bc62-153b79863396" width="50%"><img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/576affb7-740c-4170-8bd9-1beefd6283f2" width="50%">

教師あり学習のTutorialは以上です．お疲れさまでした．
ニューラルネットワークの可視化はニューラルネットワークのパラメータを表示して学習によって変化する様子を見れるような内容になっています．これについては，他のTutorialが終わって，興味があればやってみてもいいかもしれません．
# 参考資料
[CNNの基本](https://zero2one.jp/learningblog/cnn-for-beginners/)

[nlllossとは](https://qiita.com/y629/items/1369ab6e56b93d39e043)
