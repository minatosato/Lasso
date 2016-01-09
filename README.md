
#Lasso
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
誤差関数にL1ノルムの正則化を加えたものを **L1正則化(Lasso)** [[Tibshirani, 1996](http://statweb.stanford.edu/~tibs/lasso/lasso.pdf)]といいます．LassoはLeast absolute shrinkage and selection operatorの略らしいですね．

$$\boldsymbol{S}_{\lambda}(\boldsymbol{\beta})  = ||\boldsymbol{y}-\boldsymbol{X\beta}||^2 + \lambda||\boldsymbol{\beta}||_1$$

L2正則化のときと同様に， $\boldsymbol{\beta}$  で偏微分して．．．推定量を求めたいところですが，L1の誤差関数の正則化項( $i.e., \lambda |\boldsymbol{\beta}|$ )が $\boldsymbol{\beta}$ で偏微分不可能なため，L2正則化のときのようには推定量を求めることができません．そこで今回は**CD(Coordinate Descent)**[J Friedman et al., [2007](http://arxiv.org/pdf/0708.1485.pdf);[2010](http://core.ac.uk/download/files/153/6287975.pdf)]というアルゴリズムを用いて $\boldsymbol{\beta}_{lasso}$ を推定したいと思います．

$$
\newcommand{\argmin}{\mathop{\rm arg~min}\limits}
\begin{aligned}
\boldsymbol{\beta}_{lasso} & = \argmin_{\boldsymbol{\beta} \in \mathcal{R^p}}　\biggl[ ||\boldsymbol{y}-\boldsymbol{X\beta}||^2 + \lambda ||\boldsymbol{\beta}||_1 \biggr]
\end{aligned}
$$

## なぜL1正則化か
その前に，L1正則化をすると何が嬉しいのか，どういった回帰係数($\boldsymbol{\beta}$)が得られるのか確認してましょう．
L1正則化を行うと，推定したパラメータ(今回は回帰係数である$\boldsymbol{\beta}$)の一部が0になり，スパースな解が得られます．特徴量がものすごくたくさんあるとき，つまり重回帰モデルおける説明変数がものすごくたくさんあるとき，**それらの一部を0にすることによって，変数選択(特徴選択)してくれていることになります**．
変数選択する指標として，

- AIC
- BIC
- 各変数ごとの有意性検定

などが挙げられます．従来ではモデルの推定と変数選択は別々に行っていました．
Lassoでは**モデルの推定と変数選択を同時に行ってくれます．**

次に**どうしてLassoはスパースな解が得られるか**ということですが，次のサイトが参考になるかと思われます．

- [【機械学習】LPノルムってなんだっけ？ - Qiita](http://qiita.com/kenmatsu4/items/cecb466437da33df2870)
- [RでL1 / L2正則化を実践する - 東京で働くデータサイエンティストのブログ](http://tjo.hatenablog.com/entry/2015/03/03/190000)

## CDによる推定
### アルゴリズム
CDについては，

- [A Coordinate Descent Algorithm for the Lasso Problem - Jocelyn T. Chi](http://www.jocelynchi.com/a-coordinate-descent-algorithm-for-the-lasso-problem/)
- [lecture 24: coordinate descent - YouTube](https://www.youtube.com/watch?v=Mbnd5nisFNw) (カーネギーメロン大学の凸最適化の講義動画)

が非常に参考になるかと思われます．また，他の推定アルゴリズムとして有名なものは，

- Least angle regression(LARS)[[Efron et al, 2004](http://statweb.stanford.edu/~imj/WEBLIST/2004/LarsAnnStat04.pdf)]

などがあります．

CDでは，各パラメータ($i.e.,\ \beta_1, \beta_2,...,\beta_p$)毎に誤差関数を微分して更新式を得て，それを用いて更新を繰り返し行うことにより，収束した最適な推定値得ることができます．

```math
\begin{aligned}
\boldsymbol{S}_{\lambda}(\boldsymbol{\beta}) & =\frac{1}{2n} ||\boldsymbol{y}-\boldsymbol{X\beta}||^2 + \lambda||\boldsymbol{\beta}||_1
\end{aligned}
```
今回は誤差関数を上のような式として，$\beta_j$で微分して更新式を得ます．
$\beta_j \neq 0$のとき，$\beta_j$に関して微分し，$=0$とおいて，

```math
\begin{aligned}
\frac{1}{n} \biggl[ \boldsymbol{X}_{:,j}^T\boldsymbol{X}_{:,j}\beta_j +\boldsymbol{X}_{:,j}^T(\boldsymbol{X}_{:,-j}\boldsymbol{\beta}_{-j} - \boldsymbol{y}) \biggr] + \lambda sign(\beta_j)   = 0
\end{aligned}
```

$\beta_j$について解くと，


```math
\begin{aligned}
\beta_j = \frac{1}{\boldsymbol{X}_{:,j}^T\boldsymbol{X}_{:,j}} S\biggl(\boldsymbol{X}_{:,j}^T(\boldsymbol{y} - \boldsymbol{X}_{:,-j}\boldsymbol{\beta}_{-j}), \lambda n \biggr) \cdots (**)
\end{aligned}
```

ここで，$S$は**soft-thresholding operator**と呼ばれ，

```math
\begin{aligned}
S(x, \lambda) =
  \left\{
    \begin{array}{l}
      x - \lambda & (x > \lambda) \\
      0 & (|x| \leq \lambda) \\
      x + \lambda & (x < -\lambda)
    \end{array}
  \right.
\end{aligned}
```

という式で表されます．具体的には$\lambda=1$のとき，横軸$x$，縦軸$S(x, \lambda)$のグラフは次のようになります．

<img src="https://qiita-image-store.s3.amazonaws.com/0/31899/3f5836d8-d54d-f4ac-09e3-0421edd19959.png" width=480 />

また，切片(intercept)である$\beta_0$については，通常は正則化を考慮しません．2乗誤差を$\beta_0$について微分し，求める事ができ，

```math
\begin{aligned}
\beta_0 = \frac{\sum_{i=1}^{n}( y_i-\boldsymbol{X}_{i,:}\boldsymbol{\beta})}{n}\cdots (***)
\end{aligned}
```

となります．
あとは$\beta_i$の更新を$1,\cdots, p$について繰り返し行えばいいだけです！  
更新アルゴリズムの擬似コードを示します．  


LassoにおけるCoordinate Descent  
01　$\boldsymbol{\beta}$の初期化，$\lambda$の固定  
02　if 切片がある:  
03　　　$(\*\*\*)$式で$\beta_0$更新．  
04　while 収束条件を満たさない:  
05　　　for $j$ in $1,\cdots, p$:  
06　　　　　$(\*\*)$式で$\beta_j$を更新．  
07　　　if 切片がある:  
08　　　　　$(\*\*\*)$式で$\beta_0$更新．  




最後に，正則化パラメータ$\lambda$の大きさを変えたときに，得られる係数の値について見ていきたいと思います．$\lambda$を大きくすればするほど，正則化項の値が大きくなります．つまり，回帰係数$\boldsymbol{\beta}$のうち0となるものが増えていきます．逆に$\lambda$を小さくしていけば，通常の重回帰モデルと同じくなるので，回帰係数$\boldsymbol{\beta}$のうち0となるものが減っていきます．$\lambda$を変化させたときに得られる係数をプロットしてみたのが，以下の図です．

<img src="https://qiita-image-store.s3.amazonaws.com/0/31899/58f059ba-7741-5e8e-580b-e32acaee3a85.png" width=640>

うっ色が被っててわかりにくい．．．


# 最後に
今回は重回帰モデルにおけるL1正則化，Lassoについて扱いました．

## 参考
- [A Coordinate Descent Algorithm for the Lasso Problem - Jocelyn T. Chi](http://www.jocelynchi.com/a-coordinate-descent-algorithm-for-the-lasso-problem/)
- [lecture 24: coordinate descent - YouTube](https://www.youtube.com/watch?v=Mbnd5nisFNw) (カーネギーメロン大学の凸最適化の講義動画)
- [【機械学習】LPノルムってなんだっけ？ - Qiita](http://qiita.com/kenmatsu4/items/cecb466437da33df2870)
- [RでL1 / L2正則化を実践する - 東京で働くデータサイエンティストのブログ](http://tjo.hatenablog.com/entry/2015/03/03/190000)
- [Regularization Paths for Generalized Linear Models via Coordinate Descent](http://core.ac.uk/download/files/153/6287975.pdf)
- [PATHWISE COORDINATE OPTIMIZATION](http://arxiv.org/pdf/0708.1485.pdf)








```math
\begin{aligned}

\end{aligned}
```
