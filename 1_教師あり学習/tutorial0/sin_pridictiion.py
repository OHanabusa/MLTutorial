import matplotlib.pyplot as plt
import numpy as np
import os

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
plt.rcParams["figure.figsize"] = [6,4]
plt.rcParams["xtick.major.pad"] = 10
plt.rcParams["figure.dpi"] = 120  # need at least 300dpi for paper
plt.rcParams["figure.autolayout"] = True

#出力の保存先の作成
folder_name = os.path.splitext(os.path.basename(__file__))[0]#実行しているファイルの名前
folder_dir = os.path.dirname(os.path.abspath(__file__))#実行しているファイルのパス（どこにあるか）
# dr = folder_name+"/results" #+'/'+file_name
dr = folder_dir + "/"+ folder_name + "_results"
os.makedirs(dr, exist_ok=True)#ファイルを作成

# 訓練データ作成
x_train = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)  # 入力値を(100, 1)の形状に変更
y_train = np.sin(x_train)  # 出力値(sin(x))
x_test = np.linspace(-2*np.pi, 2*np.pi, 99).reshape(-1,1)

# ネットワーク構造の設定
input_size = 1  # 入力層のノード数(=1次元のxの値)
hidden_size = 10  # 隠れ層のノード数
output_size = 1  # 出力層のノード数(=sin(x)の値)

# 重み行列とバイアスの初期化
W1 = np.random.randn(input_size, hidden_size)  # 入力->隠れ層の重み
b1 = np.zeros(hidden_size)  # 隠れ層のバイアス
W2 = np.random.randn(hidden_size, output_size)  # 隠れ->出力層の重み  
b2 = np.zeros(output_size)  # 出力層のバイアス

# 活性化関数(シグモイド関数)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 前向き伝搬
def forward(x):
    global a1
    z1 = np.dot(x, W1) + b1  # 入力層->隠れ層
    a1 = sigmoid(z1)  # 隠れ層の活性化
    z2 = np.dot(a1, W2) + b2  # 隠れ層->出力層
    return z2  # 出力層の値(=予測値)

# 損失関数(平均二乗誤差)
def loss(y_pred, y_true):
    return np.mean((y_true - y_pred)**2)

# 勾配計算
def backprop(x, y_true):
    global a1
    y_pred = forward(x)
    
    # 出力層の勾配
    grad_z2 = 2 * (y_pred - y_true) / x.shape[0]
    grad_W2 = np.dot(a1.T, grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0)
    
    # 隠れ層の勾配
    grad_a1 = np.dot(grad_z2, W2.T)
    grad_z1 = grad_a1 * a1 * (1 - a1)
    grad_W1 = np.dot(x.T, grad_z1)
    grad_b1 = np.sum(grad_z1, axis=0)
    
    return grad_W1, grad_b1, grad_W2, grad_b2

# 確率的勾配降下法による重み・バイアスの更新
learning_rate = 0.01
EPOCHS = 50000
for epoch in range(EPOCHS):
    grad_W1, grad_b1, grad_W2, grad_b2 = backprop(x_train, y_train)
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2
    
    # 損失の表示
    if epoch % (EPOCHS//10) == 0 or epoch == EPOCHS-1:
        y_pred = forward(x_train)
        loss_val = loss(y_pred, y_train)
        print(f'Epoch {epoch}, Loss: {loss_val:.4f}')
        
# 予測値の計算とプロット
        y_pred = forward(x_test)
        fig = plt.figure()
        plt.plot(x_train, y_train, label='True')
        plt.plot(x_test, y_pred, label='Predicted')
        plt.legend(fontsize=14, loc="upper right")
        fig.savefig(dr+f"/epoch{epoch}_sin_pridict.png") 
        plt.close()      
