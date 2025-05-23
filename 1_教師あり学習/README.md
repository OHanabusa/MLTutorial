ここではニューラルネットワークを用いた教師あり学習の理論を紹介する．その後，理論とプログラムを比較しながら実際にプログラムを動かし，学習が行われる様子を体験してもらう．
# 教師あり学習の概要
まず，教師あり学習では，入力データとその正解のラベルが分かっている必要がある．例えば，猫と犬の画像からどちらか認識するニューラルネットワークを学習するときは，それぞれの画像に正解データとして犬または猫という情報が紐づいている必要がある．
<p align="center">
  <img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/a5b3ede2-c06e-4a05-b5f8-4a8154c61148" width="50%">
</p>

ニューラルネットワークを用いた教師あり学習における学習の手順は以下の通りである．
1. __順伝播（Forward Propagation）__：ニューラルネットワークにおいて情報を入力した際，入力層から出力層へ順方向として出力が計算される．各層の出力は，入力とバイアス・重み行列と呼ばれる変数を行列計算し，活性化関数に入力することで計算される．
2. __誤差の計算__：損失関数を用いて，出力と正解の誤差が計算される．また，損失関数は学習モデルの性能を評価するための関数でもある．
3. __逆伝播（Backward Propagation）__：出力層から入力層に向かって順伝搬とは逆方向に，各層の誤差が計算される．各層の誤差は，その層の重み行列と活性化関数の勾配に基づいて計算される．さらに，各ノードの誤差は，直前の層に伝播される
4.  __勾配の計算__：各層での誤差を用いて，各バイアス・重みの勾配が計算される．勾配は，損失関数を各重みで偏微分することで計算される．この際，連鎖則を用いて，誤差がどのように前の層に伝播するか計算される．
5.  __バイアス・重みの更新__：勾配降下法やその変種を用いて，各バイアス・重みが更新される．

以上の手順を各入力データに対して行うことによって，学習が進み，出力が正解に近づく．
# 教師あり学習の理論
## ニューラルネットワークの構造
<p align="center">
  <img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/34174a6b-9629-46f9-874c-8fd25867b128">
</p>

ニューラルネットワークは入力層，出力層，そしてその間にある複数の隠れ層から構成される．入力層と出力層は問題に応じて必要なノード数を決める．隠れ層のノード数は自由に決めることができる．上図に示すように，入力層には入力ベクトル，入力ベクトルから全結合された隠れ層には隠れ層ベクトル，隠れ層ベクトルから全結合された出力層には出力層ベクトルが各層のノードの情報となる．
入力層には入力ベクトル  $\boldsymbol x$  の次元数のノードが存在し，各要素の実数を保持する．
隠れ層や出力層の各層には，複数のノードが存在する．このとき，第 $l$ 層の $i$ 番目のノードを $h_i^{(l)}$ と表記する．
各ノード $h_i^{(l)}$ は，隠れ層の初めの層を1，出力層を $L$ とする層番号 $l(=1,2,...,L)$ ，第 $l-1$ 層の $i$ 番目のノードと第 $l$ 層の $j$ 番目のノードを結ぶ重み行列 $W_{ij}^{(l)}$ ，第 $l$ 層の $i$ 番目のノードのバイアス項 $b_i^{(l)}$ ，第 $l$ 層の活性化関数 $f^{(l)}$ (ReLU, tanhなど)を用いて，以下のように定義される．

$$
\begin{align}
  (1)&&h_i^{(l)} &= f^{(l)} I_i^{(l)}\\
  (2)&&I_i^{(l)} &= \sum_j W_{ij}^{(l)} h_j^{(l-1)} + b_i^{(l)}
\end{align}
$$

したがって，各ノードでは，1つ前の層の出力と重み行列の積の合計にバイアスを加え，活性化関数を通して，各ノードの値が計算される．これを行列形式で表現すると以下のように表される．

$$
\begin{equation}
  (3)\quad \boldsymbol h^{(l)} =
  f\left(
  \boldsymbol {W}^{(l)} \cdot \boldsymbol {h}^{(l-1)} + \boldsymbol {b}^{(l)}
  \right)
\end{equation}
$$

実際のプログラムでニューラルネットワークの順伝搬の計算を行う際はこのような行列計算を行う．

また，この重み行列 $\boldsymbol W^{(l)}$ やバイアスベクトル $\boldsymbol b^{(l)}$ がニューラルネットワークの学習パラメータであり，第 $l$ 層目の重み行列 $\boldsymbol W^{(l)}$ とバイアスベクトル $\boldsymbol b^{(l)}$ の初期値は，一般的に，第 $l-1$ 層目のノード数 $d$ に対して， $\left(-\frac{1}{\sqrt{d}}, \frac{1}{\sqrt{d}}\right)$ の範囲でランダムに設定される．

## 活性化関数
活性化関数は，非線形な関数の学習を可能にするという役割を果たす，ニューラルネットワークにおける重要な構成要素である．そのため，活性化関数を取り入れることでニューラルネットワークの表現力を向上させることができる．また，活性化関数には，いくつか種類があり，代表的なものとしてステップ関数，sigmoid関数，ReLU関数などが挙げられる．これらはその関数形や学習するデータに応じて適切な役割が決まっており，用途によって使い分けられる．一般的に，隠れ層の活性化関数はReLU関数が使われ，出力層の活性化関数は得たい出力に応じて決める．-1～1の範囲の出力が欲しいのならばtanh，0～1の範囲であればsigmoidという感じである．

例として，tanhとLeaky ReLU関数を以下に示す．

$$
\begin{align}
    f_{\tanh} &=  \frac{e^{2x} - 1}{e^{2x} + 1}\\
    f_{\mathrm{Leaky ReLU}} &= \begin{cases}
    x, & \text{if } x > 0 \\
    \alpha x, & \text{if } x \leq 0
    \end{cases}
\end{align}
$$

ここで， $\alpha$ はLeaky ReLUの傾きを示し，一般的に0.01とされる．これらの関数をプロットした様子を下図に示す．図に示すように，tanhは出力が-1から1の範囲に限定されることが分かる．また， $x=0$ 付近で勾配が大きいため学習効率が高いが， $x=0$ から離れた領域では勾配がゼロになるという特徴がある．Leaky ReLUは $x<0$ において出力をゼロとするReLU関数の変種である．そのため， $x<0$ 領域において微小な傾きを持ち，勾配が全領域でゼロにならないという特徴がある．
<p align="center">
  <img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/652cbb6d-ad5b-41b2-ba7e-4edffd02d8db" width="60%">
</p>

その他の活性化関数は参考資料のサイトまたは各自で調べてみてください．活性化関数は出力の範囲を定めるだけでなく，活性化関数の勾配が学習を行う際にとても重要な要素となっているのでいろいろ試してみてください．

## 誤差関数
損失関数（loss function）は，機械学習モデルがどれくらい正しく予測できているか評価する指標です．また，ニューラルネットワークの学習において損失関数を最小化するようにパラメータを更新します．損失関数は任意の関数を用いることができますが，クラス分類や値の予想などの問題によってそれぞれ一般的に使われる関数があります．最もよく使われる関数が2乗和誤差と呼ばれ，以下のように定義される．

$$
E =  \sum_i \frac{1}{2}({y}_i - {t}_i)^2
$$

ここで， $y_i$ は $i$ 番目の出力， $t_i$ が正解データである．これは値の予測や線形回帰問題に使われる関数で，各出力が正解データに近いほど誤差は小さくなる．

## 誤差逆伝搬法
<p align="center">
  <img src="https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/assets/130419605/fada9334-c13b-4c08-ba55-0bb1e2ecb870">
</p>
誤差逆伝播法は，ニューラルネットワークの学習パラメータを最適化するために誤差を出力層から入力層に向けて逆方向に伝播させながら損失関数を最小化するように各層の重みとバイアスを更新するアルゴリズムである．

第 $l$ 層のバイアスと重みを更新するには，損失関数の偏微分を連鎖則を用いて計算する必要があり，最終的に，以下の式で定義できる（詳しくは誤差逆伝搬法を参照してください．）

$$
\begin{align}
  (4)&&\frac{\partial E}{\partial \boldsymbol h^{(l)}} &=  \begin{cases} \frac{\partial E}{\partial \boldsymbol h^{(L)}} & ({\rm{if}} \quad l = L)\\
          \left(\boldsymbol W ^{(l+1)}\right)^T \cdot 
  \frac {\partial E}{\partial \boldsymbol b^{(l+1)}} &(\rm otherwise)
      \end{cases}\\
  (5)&&\frac {\partial E}{\partial \boldsymbol b^{(l)}} 
      &= 
  \frac {\partial E}{\partial \boldsymbol h^{(l)}} \circ 
  \frac {\partial \boldsymbol h^{(l)}}{\partial \boldsymbol I^{(l)}} \\
  (6)&&\frac {\partial E}{\partial \boldsymbol W^{(l)}}
      &=  \boldsymbol h ^{(l-1)} \cdot \left(
  \frac {\partial E}{\partial \boldsymbol b^{(l)}} \right)^T
\end{align}
$$

ここで， $\circ$ は同じ要素の積で計算されるアダマール積である．また， $\frac {\partial \boldsymbol h^{(l)}}{\partial \boldsymbol I^{(l)}}$ は式（1），（2）からわかるように，活性化関数をその入力で偏微分したものである．
式（4）～（6）は出力層から順番に式（4），式（5）式（6）と繰り返し計算することですべての層の勾配が得られることが分かる．出力層から入力層に向けて誤差を伝搬させていることから誤差逆伝搬法と呼ばれている．
これを用いて，重み・バイアスのパラメータを以下の式で更新する．

$$
\begin{align}
  (6)&&\boldsymbol {b}^{(l)} &\leftarrow \boldsymbol {b}^{(l)} - \eta \frac{\partial E}{\partial \boldsymbol {b}^{(l)}} \\
  (7)&&\boldsymbol {W}^{(l)} &\leftarrow \boldsymbol {W}^{(l)} - \eta \frac{\partial E}{\partial \boldsymbol {W}^{(l)}} 
\end{align}
$$

ここで， $\eta$ は学習率(learning rate)と呼ばれる定数であり，更新における変化量を制御している．この更新式によって，学習パラメータ $\boldsymbol W^{(l)}$ ， $\boldsymbol b^{(l)}$ を損失関数 $E$ が最小化する方向に更新される．また，この更新式は最急降下法(SGD)と呼ばれ，最も単純な更新式である．これ以外にも前回の変化量を考慮したものや，変化に慣性力を加えたものなどたくさんあるのでいろいろ試してみてください．特に優秀な更新式としてAdamと呼ばれるものがあります．

## プログラムを動かしてみよう
<<<<<<< HEAD
次は，1_教師あり学習/tutorial0 に進んでプログラムを実際に動かして理解しましょう．
=======
次は， [__tutorial0__](https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/tree/6fa0124831dee20f2b9331f744e98c67a299aefe/1_%E6%95%99%E5%B8%AB%E3%81%82%E3%82%8A%E5%AD%A6%E7%BF%92/tutorial0) に進んでプログラムを実際に動かして理解しましょう．
>>>>>>> f1678cbddafc38a73e5719597ba54a2e46566e7f

## 補足資料
[英の卒論](https://github.com/SolidMechanicsGroup/ML_Tutorial_2024/blob/33ce72255cbbb695ec96588a9e1aa9ab11727390/%E5%8D%92%E6%A5%AD%E8%AB%96%E6%96%87_%E8%8B%B1%E9%9F%B3.pdf)

[ニューラルネットワークとは](https://udemy.benesse.co.jp/data-science/ai/neural-network.html)

[活性化関数](https://nisshingeppo.com/ai/activation-functions-list/)

[損失関数](https://www.tech-teacher.jp/blog/loss-function/)

[誤差逆伝搬法](https://qiita.com/43x2/items/50b55623c890564f1893#%E4%B8%80%E8%88%AC%E5%8C%96)

[最適化法](https://www.tech-teacher.jp/blog/algorithm-machine-learning/)

[東工大の資料](http://www.iee.e.titech.ac.jp/~nakatalab/text/lib/pdf/kougisiryou/ml/ML5.pdf)
