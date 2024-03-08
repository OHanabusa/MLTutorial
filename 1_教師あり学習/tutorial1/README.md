# MNIST手書き数字を全結合型ニューラルネットワークで処理してみよう
ここでは，[このサイト](https://atmarkit.itmedia.co.jp/ait/articles/2005/21/news017.html)で紹介されていたニューラルネットワークを用いた手書き文字のクラス分類を行う．説明でわからないところがあれば，サイトを参照してみてください．

## プログラムの準備
1. まず初めにhandwriting.pyとrequirement.txtをvscodeやほかのpython環境にダウンロードしてください．
2. その後，ターミナルで`pip install -r requirements.txt`と入力することで必要なモジュールをダウンロードします．
この時，`pip install -r ???`の???部分はダウンロードしたrequirements.txtのパスを入力して下さい．
3. `python handwriting.py`で実行できると思います．プログラムと以下の説明を比較しながら理解してください．

## プロクラムの説明
MNISTデータベースには、下に示したような手書きの数字（と対応する正解ラベル）が訓練データとして6万個、テストデータとして1万個格納されています。この膨大な数のデータを使用して、ニューラルネットワークを用いた手書き数字の認識をしてみようというのが目標です。

![image](https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/09e2a68a-fbde-4237-ac96-708b36455c59)

まず，下のコードでは，訓練データ(trainloader)とテストデータ(testloader)に分けて定義します．
手書き文字の画像は28×28のピクセルデータがそれぞれ0～255のグレースケールの値で保存されているため，読み込む際に`transform`を用いて0～１に正規化します．

```python
BATCH_SIZE = 20
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.0,), (1.0,))])

train_set = torchvision.datasets.MNIST(root='./', train=True,transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./', train=False,transform=transform, download=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
```

次は，ニューラルネットワークの層の数とそれぞれの活性化関数の定義をしています．ここではpytorchという機械学習のモジュールを使用しています．ほかにもTensorflowやJaxなどがあるので，好きなものを利用してください．pytorchのコードの引数などの詳細は[pytorch公式](https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html#torch.nn.functional.linear)で調べてください．

ニューラルネットワークの構造は入力層，１つの隠れ層，出力層のノード数がそれぞれ`INPUT_FEATURES`, `HIDDEN`, `OUTPUT_FEATURES`個となっている．今回は入力層のノードを28×28，出力層を10個とする．出力層のノードはそれぞれ0～9の数字の分類に対応している．また，隠れ層の層の数やノード数は任意に設定できます．活性化関数は隠れ層のReLU関数のみである．出力層の活性化関数としてクラス分類で使われるsoftmaxを使ってもよい．

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

損失関数は交差エントロピー関数，最適化法は学習率lr=0.001のSGDとしていますが他のも試してみてください．最適化法によって学習の成績がかなり変わります．

```python
#損失関数の設定
criterion = torch.nn.CrossEntropyLoss()
#最適化法の設定
#通常のSGD
optimizer = optim.SGD(net.parameters(), lr=0.001)
```

以上がデータとニューラルネットワークの設定で，以降は学習になります．

ここでは，訓練データを使って学習を行っています．`outputs`がニューラルネットワークの出力で，`loss`が損失関数の計算，`loss.backward()`で勾配の計算，`optimizer.step()`でニューラルネットワークの更新を行っています．これは，教師あり学習の理論で説明した勾配の計算とニューラルネットワークの更新を自動でやってくれています．

```python
EPOCHS = 10#データセットを何周するか
for epoch in range(1, EPOCHS + 1):
    #学習フェーズ
    for count, item in enumerate(trainloader, 1):
        inputs, labels = item
        inputs = inputs.reshape(-1, 28 * 28)#28*28の２次元配列を１次元の配列に変換
        optimizer.zero_grad()
        outputs = net(inputs)#Netクラスのforward関数が使われる
        loss = criterion(outputs, labels)
        #ニューラルネットワークの更新
        loss.backward()
        optimizer.step()
```

次は，１セットの訓練データが学習し終わったタイミングでテストデータを使って現在のニューラルネットワークの成績を確認します．

ここでは，ニューラルネットワークの更新は行わないため，`torch.no_grad()`で勾配計算を行わないモードに変更しています．ここでは，`predicted`でニューラルネットワークが予測した分類を取得し，`correct += (predicted == labels).sum().item()`で正解数を数えています．

```python
    with torch.no_grad():#勾配が計算されないモード
        for data in testloader:
            inputs, labels = data
            inputs = inputs.reshape(-1, 28 * 28)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)#出力層の最大値を示すものをニューラルネットワークの予測した分類とする．
            total += len(outputs)
            correct += (predicted == labels).sum().item()
```

以上の学習と評価を`EPOCHS`回繰り返し行い，その学習の推移をグラフにプロットします．

以下に，最適化法Adam，EPOCHS=30としたときの結果を示す．

<p align="center">
  <img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/a7633edd-fed3-4a16-8f57-ecdcc39a7abe" width="40%"><img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/9eb583f2-c84c-41a2-8339-f2e964e5588a" width="40%">
</p>

隠れ層の層の数，ノード数，活性化関数，EPOCHS，学習率，最適化法などいろんな変数を調整して精度98%を超えられるか挑戦してみてください．
