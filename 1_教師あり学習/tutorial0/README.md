# ニューラルネットワークによる関数の予測

ここでは，ニューラルネットワークに教師あり学習を用いてsin関数の近似を行います．
また，ニューラルネットワークの中身を理解するためにニューラルネットワークを行列形式で表現しています．

## プログラムの準備
1. まず初めにsin_pridiction.pyとrequirement.txtをvscodeやほかのpython環境にダウンロードしてください．
2. その後，ターミナルで`pip install -r requirements.txt`と入力することで，このtutorialを通して必要なモジュールをダウンロードします．
この時，`pip install -r ???`の???部分はダウンロードしたrequirements.txtのパスを入力して下さい．
4. `python sin_pridiction.py`で実行できると思います．プログラムと以下の説明を比較しながら理解してください．

## プログラムの説明
ニューラルネットワークの構造としては入力層，隠れ層，出力層がそれぞれ１層ずつ，各層のノード数はそれぞれ1，10，1個とした．
```python
# 訓練データ作成
x_train = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)  # 入力値を(100, 1)の形状に変更
y_train = np.sin(x_train)  # 出力値(sin(x))
# テストデータ作成
x_test = np.linspace(-2*np.pi, 2*np.pi, 101).reshape(-1,1) #学習データx_trainとは異なる値をテストデータとして定める
y_test = np.sin(x_test)
```

## 学習結果

学習回数１回

<img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/351ec4b8-d968-484d-944d-ca564b77523e" width="40%">

学習回数10万回

<img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/43bf8597-a4a6-49f1-a006-1f80e8b052e3" width="40%">

学習回数100万回

<img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/0cbab061-e610-4710-b013-3d0ae8627188" width="40%">


