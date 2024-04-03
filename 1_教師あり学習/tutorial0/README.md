# ニューラルネットワークによる関数の予測

ここでは，ニューラルネットワークに教師あり学習を用いてsin関数の近似を行います．
また，ニューラルネットワークの中身を理解するためにニューラルネットワークを行列形式で表現しています．

## プログラムの準備
1. まず初めにsin_pridiction.pyとrequirement.txtをvscodeやほかのpython環境にダウンロードしてください．
2. その後，ターミナルで`pip install -r requirements.txt`と入力することで，このtutorialを通して必要なモジュールをダウンロードします．
この時，`pip install -r ???`の???部分はダウンロードしたrequirements.txtのパスを入力して下さい．
4. `python sin_pridiction.py`で実行できると思います．プログラムと以下の説明を比較しながら理解してください．

## プログラムの説明
まず初めに，訓練データとテストデータの作成を行う．訓練データを用いてニューラルネットワークの学習を行い，学習データとして使っていない未知のテストデータを用いて関数近似の精度を評価する．

```python
# 訓練データ作成
x_train = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)  # 入力値を(100, 1)の形状に変更
y_train = np.sin(x_train)  # 出力値(sin(x))
# テストデータ作成
x_test = np.linspace(-2*np.pi, 2*np.pi, 101).reshape(-1,1) #学習データx_trainとは異なる値をテストデータとして定める
y_test = np.sin(x_test)
```

次に，ニューラルネットワークを作成する．
ニューラルネットワークの構造としては入力層，隠れ層，出力層がそれぞれ１層ずつ，各層のノード数をそれぞれ1，10，1個とした．
また，重み行列の初期値は $\left(-\frac{1}{\sqrt{d}}, \frac{1}{\sqrt{d}}\right)$ ，バイアスの初期値は0とした．
```python
# ネットワーク構造の設定
input_size = 1  # 入力層のノード数(=1次元のxの値)
hidden_size = 10  # 隠れ層のノード数
output_size = 1  # 出力層のノード数(=sin(x)の値)

# 重み行列とバイアスの初期化
W1 = np.random.normal(0, pow(input_size, -0.5), size=(input_size, hidden_size))  # 入力->隠れ層の重み
b1 = np.zeros(hidden_size)  # 隠れ層のバイアス
W2 = np.random.normal(0, pow(hidden_size, -0.5), size=(hidden_size, output_size))  # 隠れ->出力層の重み  
b2 = np.zeros(output_size)  # 出力層のバイアス
```

次に，ニューラルネットワークの順伝搬を定義する．理論で説明した行列計算をnumpyのドット積で行う．また，活性化関数は1層目のシグモイド関数のみで，出力層の活性化関数はなしとした．
```python
# 活性化関数(シグモイド関数)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 順伝搬
def forward(x):
    global a1
    z1 = np.dot(x, W1) + b1  # 入力層->隠れ層
    a1 = sigmoid(z1)  # 隠れ層の活性化
    z2 = np.dot(a1, W2) + b2  # 隠れ層->出力層
    return z2  # 出力層の値(=予測値)
```



## 学習結果
学習過程を以下に示す．
```python
Epoch 0, Loss: 0.6472
Epoch 100000, Loss: 0.0796
Epoch 200000, Loss: 0.0645
Epoch 300000, Loss: 0.0609
Epoch 400000, Loss: 0.0522
Epoch 500000, Loss: 0.0396
Epoch 600000, Loss: 0.0056
Epoch 700000, Loss: 0.0026
Epoch 800000, Loss: 0.0013
Epoch 900000, Loss: 0.0007
Epoch 999999, Loss: 0.0005
```
各学習過程における関数近似の様子を以下に示す．

学習回数１回

<img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/351ec4b8-d968-484d-944d-ca564b77523e" width="40%">

学習回数10万回

<img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/43bf8597-a4a6-49f1-a006-1f80e8b052e3" width="40%">

学習回数100万回

<img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/0cbab061-e610-4710-b013-3d0ae8627188" width="40%">


