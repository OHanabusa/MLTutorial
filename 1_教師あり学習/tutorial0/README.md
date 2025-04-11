# ニューラルネットワークによる関数の予測

ここでは，ニューラルネットワークに教師あり学習を用いて $\sin(x)$ 関数の近似を行います．特に，ネットワークの**構造**と**学習の仕組み（順伝播・誤差逆伝播）**を「行列形式」で明示的に記述することで，理論とコードの対応関係を理解することを目的としています．

---

## プログラムの準備
1. `sin_pridiction.py` と `requirements.txt` を Python 環境 (VSCode など) に保存します．
2. ターミナルで以下を実行し，必要なモジュールをインストールしてください：
   ```bash
   pip install -r requirements.txt
   ```
   `requirements.txt` のファイルパスを適宜指定してください．
3. 以下のコマンドで実行できます：
   ```bash
   python sin_pridiction.py
   ```

---

## プログラムの説明

### Step 1: データの準備

```python
x_train = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)
y_train = np.sin(x_train)
x_test = np.linspace(-2*np.pi, 2*np.pi, 101).reshape(-1, 1)
y_test = np.sin(x_test)
```

- **入力** `x` は $[-2\pi, 2\pi]$ の範囲を等間隔にサンプリングしたベクトル．
- **出力** `y = sin(x)` を教師データとすることで，「関数の形」を学習させます．
- テストデータは訓練データと異なる点でサンプリングし，汎化性能を評価します．

---

### Step 2: ネットワークの初期化

```python
W1 = np.random.normal(0, pow(input_size, -0.5), size=(1, 10))
b1 = np.zeros(10)
W2 = np.random.normal(0, pow(hidden_size, -0.5), size=(10, 1))
b2 = np.zeros(1)
```

ここでは以下のようなネットワーク構造です：

- **入力層**: $x \in \mathbb{R}^1$
- **隠れ層**: $a_1 = \sigma(W_1 x + b_1)，W_1 \in \mathbb{R}^{1 \times 10}$
- **出力層**: $\hat{y} = W_2 a_1 + b_2，W_2 \in \mathbb{R}^{10 \times 1}$

活性化関数 $\sigma(x)$ はシグモイド関数：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

---

### Step 3: 順伝播（Forward Propagation）

```python
def forward(x):
    global a1
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    return z2
```

この処理により，入力 $x$ に対して予測値 $\hat{y}$ を出力する関数が定義されます．

### 🔹 なぜ出力層に活性化関数がないのか？
これは回帰問題（ここでは $sin(x)$ の予測）であるためです。

回帰では、出力が連続値（任意の実数）を取る必要があります。

出力層に活性化関数（例：シグモイド、ReLUなど）を入れてしまうと、出力がその関数の定義域・値域に制限されてしまう。

活性化関数	出力の範囲	回帰問題に不適な理由
Sigmoid	(0, 1)	$\sin(x)$ のような $[-1, 1]$ の関数を近似できない
Tanh	(-1, 1)	範囲が制限されてしまう（まだマシだが精度に影響）
ReLU	$[0, \infty)$	負の値が出せない
したがって、「出力層は線形（活性化なし）」のままにしておくのが、回帰問題では一般的です.

---

### Step 4: 損失関数（Mean Squared Error）

```python
def mean_squared_error(y, t):
    return np.mean((y - t)**2)
```

これは以下の損失を最小化することに対応：

$$
L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}^{(i)} - t)^2
$$

---

### Step 5: 誤差逆伝播（Backpropagation）

```python
def backprop(x, t, lr):
    global W1, b1, W2, b2
    y = forward(x)
    batch_size = x.shape[0]

    # 出力層の勾配
    dy = (y - t) * 2 / batch_size
    dW2 = np.dot(a1.T, dy)
    db2 = np.sum(dy, axis=0)

    # 隠れ層の勾配
    da1 = np.dot(dy, W2.T)
    dz1 = da1 * (1 - a1) * a1
    dW1 = np.dot(x.T, dz1)
    db1 = np.sum(dz1, axis=0)

    # パラメータの更新
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
```

#### 🔹 誤差逆伝播法の式の導出（ステップバイステップ）

### 🔸 構造の前提

- 入力 $x \in \mathbb{R}^1$
- 隠れ層の出力 $a_1 = \sigma(z_1)$，ここで $z_1 = W_1 x + b_1$
- 出力 $\hat{y} = z_2 = W_2 a_1 + b_2$
- 正解ラベル $t = \sin(x)$
- 損失関数（平均二乗誤差）：

$$
L = \frac{1}{2} (\hat{y} - t)^2
$$

（※微分しやすくするために $\frac{1}{2}$ を付けています）

---

### 🔸 Step 1: 出力層の勾配

まず，損失 $L$ を出力 $\hat{y}$ に関して微分：

$$
\frac{\partial L}{\partial \hat{y}} = \hat{y} - t
$$

出力 $\hat{y} = z_2 = W_2 a_1 + b_2$ より，

- 勾配 $\frac{\partial L}{\partial W_2}$ は：

$$
\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W_2} = (\hat{y} - t) \cdot a_1^\top
$$

- 勾配 $\frac{\partial L}{\partial b_2}$ は：

$$
\frac{\partial L}{\partial b_2} = \hat{y} - t
$$

---

### 🔸 Step 2: 隠れ層の勾配

まず，出力層から隠れ層への誤差信号（デルタ）を定義：

$$
\delta_2 = \frac{\partial L}{\partial \hat{y}} = \hat{y} - t
$$

これを使って隠れ層に伝播：

$$
\delta_1 = \left( \delta_2 \cdot W_2^\top \right) \odot \sigma'(z_1)
$$

ここで $\odot$ は要素ごとの積（Hadamard積）であり、 $\sigma(z_1)$  がシグモイド関数であることから：

$$
\sigma'(z_1) = \sigma(z_1)(1 - \sigma(z_1)) = a_1(1 - a_1)
$$

よって：

$$
\delta_1 = \left( \delta_2 \cdot W_2^\top \right) \odot a_1 \odot (1 - a_1)
$$

この $\delta_1$ を使って：

- 勾配 $\frac{\partial L}{\partial W_1}$：

$$
\frac{\partial L}{\partial W_1} = x^\top \cdot \delta_1
$$

- 勾配 $\frac{\partial L}{\partial b_1}$：

$$
\frac{\partial L}{\partial b_1} = \delta_1
$$

---

### 🔸 Step 3: パラメータ更新（勾配降下法）

学習率 $\eta$ によって各パラメータを以下のように更新：

```python
W1 -= eta * dW1
b1 -= eta * db1
W2 -= eta * dW2
b2 -= eta * db2
```

---

## 🔹 全体の流れのまとめ（式一覧）

| 項目           | 数式                                                   |
|----------------|--------------------------------------------------------|
| 隠れ層の出力   | $z_1 = W_1 x + b_1,\quad a_1 = \sigma(z_1)$           |
| 出力層の出力   | $z_2 = \hat{y} = W_2 a_1 + b_2$                         |
| 損失関数       | $L = \frac{1}{2} (\hat{y} - t)^2$                       |
| 出力層の勾配   | $\frac{\partial L}{\partial W_2} = a_1^\top (\hat{y} - t)$<br>$\frac{\partial L}{\partial b_2} = \hat{y} - t$ |
| 隠れ層の誤差信号 | $\delta_1 = \left((\hat{y} - t) \cdot W_2^\top\right) \odot a_1 \odot (1 - a_1)$ |
| 隠れ層の勾配   | $\frac{\partial L}{\partial W_1} = x^\top \delta_1$<br>$\frac{\partial L}{\partial b_1} = \delta_1$ |


---

### Step 6: 学習ループ

```python
for epoch in range(EPOCHS):
    backprop(x_train, y_train, lr=0.1)
    if epoch % 100000 == 0:
        y_pred = forward(x_train)
        loss = mean_squared_error(y_pred, y_train)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

---

## 学習結果の考察

以下のように，学習が進むにつれて Loss が下がり，関数近似の精度が向上していることが確認できます：

| Epoch | Loss     |
|-------|----------|
| 0     | 0.6472   |
| 100k  | 0.0796   |
| 500k  | 0.0396   |
| 1M    | 0.0005   |

---

## 補足：なぜシグモイド？なぜ行列形式？

- **シグモイド関数**：非線形性を導入し，複雑な関数も近似できるようにする．
- **行列形式**：複数のデータに対して効率よく処理できる（バッチ学習）．勾配の導出や理論との対応が明瞭になる．

---

