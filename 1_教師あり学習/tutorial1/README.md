# MNIST手書き数字を全結合型ニューラルネットワークで処理してみよう
ここでは，[このサイト](https://atmarkit.itmedia.co.jp/ait/articles/2005/21/news017.html)で紹介されていたニューラルネットワークを用いた手書き文字のクラス分類を行う．説明でわからないところがあれば，サイトを参照してみてください．

## プログラムの準備
`handwriting.py`をダウンロードし，実行して下さい．以下の説明を比較しながら理解してください．

以下に、説明の構造と文の流れを整理して、より読みやすく・理解しやすくしたバージョンを提示します。小見出しを追加し、各コードブロックの前後に**「なぜその処理をするのか」**を明確にしています。

---

## 概要

このチュートリアルでは、**MNISTデータセット**を用いて、**手書き数字を分類するニューラルネットワーク**をPyTorchで構築・学習させます。

MNISTには以下のような28×28ピクセルの手書き数字画像が含まれており、**訓練データ6万件・テストデータ1万件**で構成されています。

<div align="center">
<img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/09e2a68a-fbde-4237-ac96-708b36455c59" width="60%">
</div>

---

### 1. データセットの準備と前処理

まず、画像データを**テンソル形式に変換**し、ピクセル値を[0, 1]の範囲に正規化します。

```python
BATCH_SIZE = 20
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,))
])

train_set = torchvision.datasets.MNIST(root='./', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
```

- `transforms.Normalize((0.0,), (1.0,))` によって単純な正規化（割るだけ）を実行。
- `shuffle=True` により、毎エポック異なる順序で訓練データを使用。

---

### 2. ニューラルネットワークの定義

#### ネットワーク構造

- **入力層**: 784ノード（28×28の画像を1次元に）
- **隠れ層**: 任意のノード数（例：100）
- **出力層**: 10ノード（数字0〜9の分類）

```python
class Net(torch.nn.Module):
    def __init__(self, INPUT_FEATURES, HIDDEN, OUTPUT_FEATURES):
        super().__init__()
        self.fc1 = torch.nn.Linear(INPUT_FEATURES, HIDDEN)
        self.fc2 = torch.nn.Linear(HIDDEN, OUTPUT_FEATURES)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)  # 活性化関数
        x = self.fc2(x)
        return x  # ソフトマックスはCrossEntropyLossに含まれる
```

> ⚠️ 出力層には`softmax`を明示的に入れていませんが、`CrossEntropyLoss`は内部で自動的にsoftmaxを使ってくれます。

---

### 3. 損失関数と最適化法の設定

```python
criterion = torch.nn.CrossEntropyLoss()  # 交差エントロピー損失
optimizer = optim.SGD(net.parameters(), lr=0.001)  # 確率的勾配降下法（SGD）
```

> 他にも`Adam`, `RMSprop`などの最適化手法があり、学習の進み具合が大きく異なります。

---

### 4. 学習ループ

#### 学習ステップ（順伝播・損失計算・逆伝播・パラメータ更新）

```python
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    for count, (inputs, labels) in enumerate(trainloader, 1):
        inputs = inputs.reshape(-1, 28 * 28)  # (batch_size, 784) に変形
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

- `loss.backward()`：自動的に各パラメータの勾配を計算。
- `optimizer.step()`：その勾配に基づいてパラメータを更新。

---

### 5. テストによる性能評価

#### 学習後のモデルでテストデータの正解率を計算

```python
    with torch.no_grad():  # 評価中は勾配を計算しない
        for inputs, labels in testloader:
            inputs = inputs.reshape(-1, 28 * 28)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # 出力の最大値を予測ラベルとする
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
```

- `torch.max(outputs, 1)` は出力テンソルの**各行で最大のインデックス（＝分類結果）**を返します。
- `correct / total` で正解率（accuracy）を計算できます。

---

### 6. 学習の可視化（オプション）

エポックごとの正解率や損失を保存しておけば、次のような学習曲線が描けます：

```python
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

---

### まとめ

このチュートリアルでは、以下の流れで手書き数字認識を体験しました：

1. **データ読み込みと前処理**
2. **ニューラルネットワークの構築**
3. **損失関数・最適化手法の設定**
4. **学習ループの実行**
5. **テストデータによる評価**

> ✨ **発展課題**：隠れ層を2層にしたり、ドロップアウト・BatchNormを追加して精度改善にチャレンジしてみてください！

---

以下に，最適化法Adam，EPOCHS=30としたときの結果を示す．

<p align="center">
  <img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/a7633edd-fed3-4a16-8f57-ecdcc39a7abe" width="40%"><img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/9eb583f2-c84c-41a2-8339-f2e964e5588a" width="40%">
</p>

隠れ層の層の数，ノード数，活性化関数，EPOCHS，学習率，最適化法などいろんな変数を調整して精度98%を超えられるか挑戦してみてください．
