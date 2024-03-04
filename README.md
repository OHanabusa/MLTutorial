# 機械学習Tutorial

## 本Tutorialの目的
ここでは，機械学習の全体像を理解し，実装できるようになることを目的とする．また，各自の研究において機械学習を用いることが有効な手段となるような場面において積極的に使うことができるようになることを望んでいる．
一方で，機械学習を行うには高いプログラミング力が求められるので，慣れないうちは実際に手を動かして，わからないことがあれば５～１０分調べてから先輩や同期に聞くことでゆっくり理解していくことが大切である．

## 機械学習の概要
機械学習とは，コンピュータに大量のデータから知識やパターンを自動的に「学習」させる技術である．機械学習アルゴリズムにデータを入力すると，アルゴリズムがデータの特徴やパターンを分析し，将来のデータに対する予測や意思決定が改善される．機械学習は人工知能の重要な構成技術の一つであり，画像認識，音声認識，自然言語処理など，様々な応用分野で活用されている．
機械学習は，大きく分類すると教師あり学習，教師なし学習，強化学習の3つに分けられる．
- 教師あり学習は，入力と正解ラベルのペアからなるデータを用いて出力が正解データに近づくように学習を行う手法であり，主にデータの分類や回帰を行う際に用いられる．近年では，ディープラーニングを用いた画像認識や自然言語処理などの分野で特に活発に利用されている．
- 教師なし学習は，入力データのみからデータの特徴や構造の学習が行われ，主なタスクはクラスタリング，次元圧縮，異常検知などである．
- 強化学習は，環境との情報のやり取りを通じて行動を決定する方策が学習される．応用例として, ゲームAIやロボット制御などが挙げられる.
また，これらの機械学習はそれぞれ，ニューラルネットワークを用いるものと用いないものがある．
ニューラルネットワークを用いない代表的な機械学習として，SVMや決定木，ベイズ推定があげられる．
一方で，ニューラルネットワークを用いた深層学習（Deep Learning）は，入力した情報から自動で学習できることが大きなメリットである．画像認識や自然言語処理など，複雑なデータを扱う課題で利用されている．強化学習でも，Q学習などのニューラルネットワークを用いない手法から，深層ニューラルネットワークを利用する深層強化学習へと発展している．
ニューラルネットワークを用いない機械学習の１つとして，

# MPMの理論
ここではMPMのアルゴリズムについての具体的なイメージを説明する. MPMはメッシュベースのものとメッシュフリーのものに大きく分けられるが, 本研究室では主にメッシュベースのものを用いるため, メッシュベースのアルゴリズムを説明する. アリゴリズムの一連の流れを次に示す[^2]. 
1. 弾性体のひずみエネルギーから, Lagrange点に生じている力を求める
2. Lagrange点上で運動方程式を解いて $\Delta t$ だけ時間発展した速度に更新する
3. APICと呼ばれる手法に基づいて, 質量と運動量をEular点へマッピングする(ここが接触処理に相当)
4. APICと呼ばれる手法に基づいて, もう一度Lagrange点へ速度をマッピングする
5. 更新された速度からLagrange点を更新する

#### 1. Lagrange点に生じた力の導出
まず, 次のような弾性体 $P$を考える. この弾性体は微小な要素に分割されており, 各要素が持っている点をLagrange点と呼ぶ.  

![diagram-20230316](https://user-images.githubusercontent.com/103396832/225524825-f9bcbfe6-2b54-47de-a6d3-95bb5ad9a644.png)

このとき, 弾性体 $P$ にひずみが生じたとすると, 点 $p$ の変位 $x_p$ を用いてひずみエネルギー $W$ を計算することができる. このひずみエネルギー $W$ を用いて次式のようにすると, Lagrange点 $p$ に生じる力 $f_p$ を求めることができる. 

$$
\begin{align}
    f_p = - \frac{\partial W}{\partial x_p}
\end{align}
$$

#### 2. Lagrange点上での速度の更新
また，Newtonの運動方程式 $m a = F$ より, 加速度 $a_p = \frac{f_p}{m_p}$ を求めることができる. この加速度から次式のようにして $\Delta t$ だけ時間発展した速度に更新できる．

$$
\begin{align}
    v_p^* = v_p + \frac{f_p}{m_p} \Delta t
\end{align}
$$

#### 3. Eular点へのマッピング
Affine Particle In Cell (APIC)と呼ばれる手法に基づいて弾性体 $P$ 上の各Lagrange点 $p$ における質量 $m_p$ と運動量 $m_p v_p^*$ をEuler点へマッピングする. 
このとき, Euler点$i$の質量, 位置, 速度をそれぞれ$m_i, x_i, v_i$とする. 
Eulerian上では弾性体の形状が存在しているわけではなく, 各Euler点が密度を持っているイメージである.   

![diagram-20230316 (1)](https://user-images.githubusercontent.com/103396832/225558659-7d2d6353-93e4-4812-b095-d917342abadc.png)

Euler点へマッピングする際に本研究室ではquadratic kernelと呼ばれる基底関数$N(x)$を用いる. 

$$
\begin{align}
    N(x) =
    \begin{cases}
    \frac{3}{4}-{|x|}^2　\quad& 0\leq|x|\leq \frac{1}{2} \\
    \frac{1}{2}\left(\frac{3}{2}-|x|\right)^2 \quad& \frac{1}{2}\leq|x|\leq \frac{3}{2} \\
    0 \quad& \frac{3}{2}\leq|x|
    \end{cases}
\end{align}
$$

<p align="center">
  <img src="https://user-images.githubusercontent.com/103396832/225817876-c0d015c2-19fd-4f43-84fc-cea9f9f3723c.png">
</p>

これを用いて重み $w_{ip}$ を次式のように与えると, Euler点 $i$ を中心としてLagrange点 $p$ までの位置に応じて値を決定する. 個人的な理解としては着目しているEuler点から離れていくほど, マッピングする際にそのLagrange点が及ぼす影響が小さくなるようなイメージである. ただし, $\Delta x$ はEuler点の間隔である. 

$$
\begin{align}
    w_{ip}&=N\left(\frac{x_i-x_p}{\Delta x}\right)N\left(\frac{y_i-y_p}{\Delta x}\right)N\left(\frac{z_i-z_p}{\Delta x}\right)
\end{align}
$$

この重み $w_{ip}$ を用いると, Euler点 $i$ 上の質量 $m_i$ , 運動量 $I_i$ および速度 $v_i$ は次式のようになる. ここで, $C_p$ はAPIC特有のaffine速度と呼ばれる補正項である. 

$$
\begin{align}
    m_i&=\sum_p w_{ip}m_p\\
    I_i&=\sum_p w_{ip} m_p \left(v_p^* + C_p  (x_i-x_p)\right)\\
    v_i&=\frac{I_i}{m_i}
\end{align}
$$

このフェーズが接触処理に当たる理由を説明する. 弾性体 $P$ と弾性体 $Q$ が接触している場合を考える. 
先ほどLagrange点が持つ速度の情報をEuler点にマッピングする際に各Lagrange点が持つ情報に重み付けして重ね合わせることでEuler点上の速度を求めた. 
弾性体同士が接触する場合は, 弾性体 $P$ 上のLagrange点の情報に加えて弾性体 $Q$ 上の情報も重み付けされて重ね合わされることになる. 

![diagram-20230317 (1)](https://user-images.githubusercontent.com/103396832/225848682-e3cc9731-a889-4639-8b86-a8860641b5e4.png)

イメージとしてはこの図のように各Euler点上で弾性体 $P$ のベクトルと弾性体 $Q$ のベクトルの合成のようなことが行われており, 接触後の挙動に合ったいい感じの向きに変わるという感じである. 
実際は全てのLagrange点からの情報が加算されてEuler点上の速度ベクトルを算出しているため, この図はあくまでイメージとして捉えてほしい.  
また, コンピュータ上では弾性体を区別して認識しているわけではなく密度の分布のような感じで存在しているため, 弾性体の数やそれぞれの状況に関わらず同じ理論で計算することができる. この弾性体を区別して認識しているわけではないというのが後に問題となるため, 頭の片隅に入れておいてほしい. 


#### 4. 再度Lagrange点へのマッピング

![diagram-20230317](https://user-images.githubusercontent.com/103396832/225839319-9cac8cd8-c42f-43df-840c-16f698151690.png)

次に求めたEuler点 $i$ 上の速度をLagrange点 $p$ にもう一度マッピングすることで $\Delta t$ 後の速度 $v_p'$ およびaffine速度 $C_p'$ は次式のように求めることができる. ただし, $d$ は基底関数の次数であり, $\langle , \rangle$はベクトル直積を表している. 

$$
\begin{align}
    v'_p&=\sum_i w_{ip} v_i\\
    C'_p&=\frac{12}{{\Delta x}^2 (d+1)} \sum_i \langle w_{ip} v_i , (x_i-x_p)\rangle
\end{align}
$$

#### 5. Lagrange点の更新
最後に次式のようにしてLagrange点 $p$ を $\Delta t$ 後の位置 $x_p'$ へ更新する. 

$$
\begin{align}
    x'_p = x_p + v'_p \Delta t
\end{align}
$$

MPMは以上で説明した一連の流れをループする. 

### 衝突修正MPM
接触判定のアルゴリズムを説明した際に接触した弾性体同士を区別できないことを述べた. 
実際の弾性体同士が衝突した際には反発力が生じるがこれまで説明してきたアルゴリズムでは弾性体同士が離れようとする際に本来存在しないはずの力が発生し, 離れられないという問題がある. 
そのため, MPMのアルゴリズムに修正を加える. 

まず, Lagrange点からEuler点にマッピングする際に接触を考慮しているが, 接触というのなら弾性体境界 $\partial \Omega$ 上にあるLagrange点のみをEuler点へマッピングする必要がある. 
そのため, 次のように修正を行った. 

$$
\begin{align}
    m_i &= \sum_{p \in \partial \Omega}  w_{ip}m_p\\
    I_i &= \sum_{p \in \partial \Omega} w_{ip} m_p \left(v_p^* + C_p  (x_i-x_p)\right)\\
\end{align}
$$

このようにして物体の境界 $\partial \Omega$ 上ではマッピング後の速度 $v_p'$ を用いて速度を更新し, 物体の内部ではマッピング前の速度 $v_p^*$ を用いて速度を更新するように修正を行った.  
また, 実際の現象において物体の接触で発生する垂直抗力の向きは必ず面法線の逆向きとなるため, 接触前後で物体表面の法線方向速度が一定または減少する場合である. この条件を面法線ベクトル $n_p$ を用いると次のようになる. 

$$
\begin{align}
    \left(v'_p -v^*_p \right) \cdot n_p >0, \quad \forall p  \in \partial \Omega
\end{align}
$$

つまり, 修正すべきなのは不適切に増加した法線方向速度であり, これを $v_p'$ の法線方向速度で置き換えると修正後の速度 $\hat{v}_p$ は次のようにできる. 


$$
\begin{align}
    \hat{v}_p = \left(v^*_p \cdot n_p\right) n_p + \left(v'_p - (v'_p \cdot n_p) n_p\right)
\end{align}
$$


したがって, 以上のことをまとめると全てのLagrange点における修正後の速度 $\hat{v}_p$ は次のように表される. 

$$
\begin{align}
    \hat{v}_p=
    \begin{cases}
        v^*_p \quad &\forall p \notin \partial \Omega\\
        \left(v^*_p \cdot n_p \right) n_p + \left(v'_p - (v'_p \cdot n_p) n_p \right) \quad &
        \{p \in \partial \Omega \mid \left(v'_p -v^*_p \right) \cdot n_p >0 \} \\
        v'_p \quad & \text{otherwise}\\
    \end{cases}
\end{align}
$$

実装する際には, Lagrangian-Eulerian間のやり取りで運動エネルギーがある程度散逸することや計算の安定性を考慮して, 法線方向速度の増加がある閾値 $\theta_p$ より大きい時に不適切な引力が働くと判断することにする. 
この閾値は経験的に $\theta_p = \left| v'_p \cdot n_p \right|$ と定める. 

衝突修正の他に接触修正という手法も存在する. 両者の違いはLagrange点上で修正を行うか, Euler点上で修正を行うかの違いであり, 衝突修正で発生する微量の誤差を改善することができる. 
ただし, ここでは取り扱わない. 

[^1]:D. Sulsky, Z. Chen, and H.L. Schreyer. A particle method for history-dependent materials. Computer Methods in Applied Mechanics and Engineering vol. 118, no. 1 (1994), pp. 179–196. doi:https://doi.org/10.1016/0045-7825(94)90112-0.

[^2]:村山太朗, Riemann多様体上の弾性論を用いたキュウリの巻きひげの力学解析, 大阪大学修士論文(2022).
