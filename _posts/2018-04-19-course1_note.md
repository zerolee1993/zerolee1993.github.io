---
layout:     post
title:      "神经网络基础"
subtitle:   "基于deeplearingai课程的学习简记（持续更新）"
date:       2018-4-19
author:     "Zero"
cover: "/assets/in-post/deeplearning.ai-1/bg.jpg"
categories: technology
tags: deeplearning
---


---

### 目录

* 纲要
{:toc}


---

### 1 认识神经网(neural network)

以房价预测为背景，用两个小例子初识神经网

例子一，面积预测售价

![png](/assets/in-post/deeplearning.ai-1/1-1.png)

抽象为单神经元网络(single neural network)，该神经元实现了ReLU

![png](/assets/in-post/deeplearning.ai-1/1-2.png)

修正线性单元ReLU(Rectified Linear Unit)，激活函数的一种

例子二，多个特征值预测售价

![png](/assets/in-post/deeplearning.ai-1/1-3.png)

将如上情况抽象成神经网络如下

![png](/assets/in-post/deeplearning.ai-1/1-4.png)

以隐层第一个神经元为例，不再代表family size，而是由神经网络自己决定

---

### 2 监督学习(supervised learning for neural network)


属于神经网络的一种，训练样本包含特征x以及明确的标签y

分为：回归(regression)、分类(classification)

一些例子：

|特征x|标签y|应用领域|神经网络类型|
|-----|-----|--------|------------|
|home features|price|房地产|standard NN|
|Ad,user info|click on ad?(0/1)|在线广告|standard NN|
|image|object(1..1000)|图像分类|CNN|
|audio|text transcript|语音识别|RNN|
|English|Chinese|机器翻译|RNN|
|Image,radar info|position of other cars|无人驾驶|hybrid NN|

---

### 3 二分类(binary classification)


通过一个识别图中是否有猫的分类任务引出一些常用的表示方式

驾驶input是一个64X64px的图片，可分解成三个颜色通道的数字矩阵，再转化为一个向量

![png](/assets/in-post/deeplearning.ai-1/1-5.png)

$x \in \mathbb{R}^{n_x}, y \in \{0, 1\}$

$m$ 表示样本的数量，$m_{train},m_{test}$ 分别表示训练集、测试集的样本数量，编码时使用"m_train"，"m_test"

$x^{(i)}$ 表示第i个样本的特征， $y^{(i)}$ 表示第i个样本的标签

训练集可以表示为 $\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),(x^{(3)},y^{(3)}),...,(x^{(m)},y^{(m)})\}$

我们习惯将x，y分别放入大的矩阵中，记为

- $X=[x^{(1)},x^{(2)},x^{(3)},...,x^{(m)}]$ , X.shape=$(n_x,m)$

- $Y=[y^{(1)},y^{(2)},y^{(3)},...,y^{(m)}]$ , Y.shape=$(1,m)$

---

### 4 逻辑回归(logistic regression)


监督学习的一种，我们希望 $y=0/1$ 或 $y\in[0,1]$ 时使用

以识别图中是否有猫为例，我们输入x，并希望得到$\hat{y}=P(y=1|x)$ , $\hat{y}$表示有猫的概率

逻辑回归中涉及到了如下参数：

- 输入： $x\in\mathbb{R}^{n_x}$
- 输出： $\hat{y}\in[0,1], \hat{y}=\sigma(w^T x + b)$ 表示图中有猫的概率
- 权重： $w\in\mathbb{R}^{n_x}$
- 偏移量： $b\in\mathbb{R}$
- 标签： $y\in 0,1$

sigmoid，公式：$s=\sigma(z)=\frac{1}{1+e^{-z}}$， 图像如下

![png](/assets/in-post/deeplearning.ai-1/1-6.png)

$z=w^T x + b $， 这一步计算是线性的，而我们希望输出在 $[0,1]$ 区间内，使用sigmoid激活函数

当 $z$ 很大时，$\sigma(z)$ 趋近于1

当 $z$ 很小时，$\sigma(z)$ 趋近于0

---

### 5 逻辑回归中的成本函数(logistic regression cost function)


在逻辑回归中，我们得到样本 $(x^{(i)},y^{(i)})$ ，希望 $\hat{y}^{(i)}\approx y^{(i)}$

#### 1、损失函数/误差函数(loss function/error function)

用于计算某一样本的误差，$L(\hat{y}^{(i)},y^{(i)})=\frac{1}{2}(\hat{y}^{(i)}-y^{(i)})^2$ 是计算误差的一个很好的方式，但是在优化过程中会导致函数非凸，梯度下降不能达到效果，所以在逻辑回归中，我们使用下面的公式

$$L(\hat{y}^{(i)},y^{(i)})=-(y^{(i)}log(\hat{y}^{(i)})+(1-y^{(i)})log(1-\hat{y}^{(i)}))$$

我们不需要纠结怎么得出的这个公式，我们看看公式效果

- 当 $y^{(i)}=1$ 时，$L=-log(\hat{y}^{(i)})$ ，若损失L减小， 则$log(\hat{y}^{(i)})$ 增大， 则$\hat{y}^{(i)}$增大，越大越趋近于1
- 当 $y^{(i)}=0$ 时，$L=-log(1-\hat{y}^{(i)})$ ，若损失L减小， 则$log(1-\hat{y}^{(i)})$ 增大， 则$1-\hat{y}^{(i)}$增大，则$\hat{y}^{(i)}$减小，越小越趋近于0

我们只要降低L，就能达到 $\hat{y}^{(i)}\approx y^{(i)}$ 的目的

#### 2、成本函数(cost function)

用于衡量在全体训练集上的表现，用每个样本的损失和的均值表示

$$J(w,b)=\frac{1}{m}\sum_{i=0}^mL(\hat{y}^{(i)},y^{(i)})=-\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(\hat{y}^{(i)})+(1-y^{(i)})log(1-\hat{y}^{(i)})]$$

---

### 6 梯度下降(gradient descent)


回顾逻辑回归的例子

$\hat{y}=\sigma(w^T x+b)$ ， $\sigma(z)=\frac{1}{1+e^{-z}}$ ， $J(w,b)=\frac{1}{m}\sum_{i=1}^mL(\hat{y}^{(i)},y^{(i)})=-\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(\hat{y}^{(i)})+(1-y^{(i)})log(1-\hat{y}^{(i)})]$

我们的目的是找到w、b，使J最小，如果w是一个标量，即只有一个特征，可以得到如下的图像

![png](/assets/in-post/deeplearning.ai-1/1-7.png)

为了直观，我们先忽略b

![png](/assets/in-post/deeplearning.ai-1/1-8.png)

梯度下降的过程就是重复 $w=w-\alpha\frac{dJ(w)}{dw}$ ，其中 $\alpha$ 表示学习率，$\alpha$ 越大，梯度下降越快

$\frac{dJ(w)}{dw}$ 为曲线的斜率，在w点的导数，编码时用"dw"表示

若起点在红点处，$dw<0$ 经过 $w=w-\alpha\frac{dJ(w)}{dw}$，w将增大，向右一步走

若起点在绿点处，$dw>0$ 经过 $w=w-\alpha\frac{dJ(w)}{dw}$，w将减小，向左一步走

重复梯度下降的过程可以使最后得到的w对应点趋近于曲线的最低点，而最低点处J最小

针对于原函数 $J(w,b)$ ,梯度下降实际上是重复如下过程

$$w=w-\alpha\frac{\partial{J(w,b)}}{\partial{w}}$$
$$b=b-\alpha\frac{\partial{J(w,b)}}{\partial{b}}$$

参数超过一个时，使用 $\partial$ 代替 $d$ ，$\partial$ 表示偏导数，编码时用"dw","db"表示

---

### 7 计算图(computation graph)


神经网络的计算过程：
- 计算神经网络输出（前向传播）
- 计算对应的梯度（反向传播）

从一个简单的例子认识计算图，设： $J(a,b,c)=3(a+bc)$

使用中间变量将计算过程拆解： $u=bc$ ， $v=a+u$ ， $J=3v$

设 $a=5$ ， $b=3$ ， $c=2$ ，先正向计算依次计算中间值和最终值，然后按虚线方向计算梯度

![png](/assets/in-post/deeplearning.ai-1/1-9.png)

反向传播详细步骤如下：
- ① 求v的梯度，$\frac{dJ}{dv}=3$ 理解为v增长1，J增长3
- ② 求a的梯度，$\frac{dJ}{da}=\frac{dv}{da}·\frac{dJ}{dv}=1·3=3$ 理解为a增长1，J增长3
- ③ 求u的梯度，$\frac{dJ}{du}=\frac{dv}{du}·\frac{dJ}{dv}=1·3=3$ 理解为u增长1，J增长3
- ④ 求b的梯度，$\frac{dJ}{db}=\frac{du}{db}·\frac{dJ}{du}=2·3=6$ 理解为b增长1，J增长6
- ⑤ 求c的梯度，$\frac{dJ}{dc}=\frac{du}{dc}·\frac{dJ}{du}=3·3=9$ 理解为c增长1，J增长9


---

### 8 逻辑回归中的梯度下降(logistic regression gradient descent)


回顾相关公式

$\hat{y}=a=\sigma(w^T x+b)$，$\sigma(z)=\frac{1}{1+e^{-z}}$，$z=w^T x+b$

单个样本损失函数 $L(a,y)=-[ylog(a)+(1-y)log(1-a)]$

全部样本成本函数 $J(w,b)=-\frac{1}{m}\sum_{i=1}^mL(a^{(i)},y^{(i)})$

#### 1、单个样本的反向传播过程

假设有两个特征值x1，x2，先通过正向传播计算出L值，并记录计算过程中的z，a的值用于反向传播

正向传播过程图略，反向传播过程如下

![png](/assets/in-post/deeplearning.ai-1/1-10.png)

我们的目标是不断更新 $w_1,w_2,b$，从而使L变小，我们先要通过反向传播计算出"dw1","dw2","db"
- ① "da" = $\frac{L}{da}=-\frac{y}{a}+\frac{1-y}{1-a}$
- ② "dz" = $\frac{dL}{dz}=-\frac{dL}{da}·\frac{da}{dz}=(-\frac{y}{a}+\frac{1-y}{1-a})·[a(1-a)]=a-y$
- ③ "db" = "dz", "dw1" = x1*"dz", "dw2" = x2*"dz"

通过反向传播我们得到了梯度值：
- dw1 = x1(a-y)
- dw2 = x2(a-y)
- db = a-y

然后我们可以进行梯度下降
- $w1 = w1 - \alpha dw1$
- $w2 = w2 - \alpha dw2$
- $b = b - \alpha db$

#### 2、m个样本的反向传播过程

$$J(w,b)=\frac{1}{m}\sum_{i=1}^{m}L(a^{(i)},y^{(i)})\text{，其中：}a^{(i)}=\hat{y}^{(i)}=\sigma(z^{(i)})=\sigma(w^T x^{(i)}+b)$$

之前讨论的针对某一样本 $(x^{(i)},y^{(i)})$，求出 $dw_{1}^{(i)},dw_{2}^{(i)},db$

在m个样本中，以 $dw_1$ 为例，实际上是所有样本的 $dw_{1}^{(i)}$ 的均值

$$\frac{\partial J}{\partial w_1}=\frac{1}{m}\sum_{i=1}^{m}\frac{\partial L}{\partial w_1}=\frac{1}{m}\sum_{i=1}^{m}dw_1^{(i)}$$

写一个算法示例(仅为了表述过程，不能执行)，实现m个样本的一次梯度下降

```python
    J = 0, dw1 = 0，dw2 = 0， db=0

    for i=1 to m:
        #正向传播
        zi = w.T * xi + b
          #这里的w是包含w1，w2的向量
          # xi表示第i个样本的两个特征组成的向量
        ai = sigmoid(zi)
        J += -[yi * log(ai)+(1-yi)log(1-ai)]
        #反向传播
        dzi = ai - yi
        dw1 += xi[0]dzi
        dw2 += xi[1]dzi # 当n= n_x时，这里也是一个for loop
        db += dzi
    J /= m, dw1 /= m, dw2 /= m, db /= m #计算均值
    #梯度下降
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    b -= learning_rate * db

```

可以看出，一次梯度下降过程中，需要使用两次显示for loop，第一次，循环m个样本，第二次，循环n_x个特征

显示调用for loop会使计算效率很低，后面我们将会用向量化来消除这两个for loop

---

### 9 向量化(vectorization)


使用矩阵运算代替for

如，当我们计算 $z=w^T x+b$ 时，$w,x\in\mathbb{R}^{n_x}$

for循环是这样计算的

    z = 0
    for i in range(n_x)
        z += w[i]* x[i]
    z += b

向量化后

    z = np.dot(w.T, x)

在使用np.function时，充分利用CPU/GPU并行化计算，SIMD计算，大大提高计算速度

来看个小例子对比两种计算的区别


```python
import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)
tic = time.time()
c = np.dot(a,b)
toc = time.time()
print("vectorization:", str(1000*(toc-tic)),'ms, value：',c)

c = 0
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()
print("for loop:", str(1000*(toc-tic)),'ms, value:',c)
```

    vectorization: 4.000425338745117 ms, value： 249913.1073508986
    for loop: 503.02863121032715 ms, value: 249913.10735088808


之前的例子是两个向量相乘，同样的方式，可以计算矩阵和向量相乘，矩阵相乘

另外np.exp()可以将矩阵的每个元素求指数，类似的还有
- np.log(v)
- np.abs(v)
- np.maximum(v)
- v ** 2
- 1/v

---

### 10 向量化实现逻辑回归前向传播


在讨论二分类时，我们将所有的x，y分别放入了一个矩阵中

下图中展示了矩阵X、Y、以及第i一个样本向量$x^{(i)}$、参数w，另外参数b是一个标量，不做展示

![png](/assets/in-post/deeplearning.ai-1/1-11.png)

未使用向量化时，正向传播过程如下

$z^{(1)}=w^T x^{(1)}+b ， a^{(1)}=\sigma(z^{(1)})$

$z^{(2)}=w^T x^{(2)}+b ， a^{(2)}=\sigma(z^{(2)})$

...

$z^{(m)}=w^T x^{(m)}+b ， a^{(m)}=\sigma(z^{(m)})$

我们将所有的 $a^{(i)}、z^{(i)}$分别放入矩阵Z、A中，得到

$A=[a^{(1)},a^{(2)} ,   ...  ,a^{(m)}]=\sigma(Z)=[\sigma(z^{(1)}),\sigma(z^{(2)}),...,\sigma(z^{(m)})]$

$Z=[z^{(1)},z^{(2)},...,z^{(m)}]=w^T X + b=[(w^T x^{(1)}+b),(w^T x^{(2)}+b),...,(w^T x^{(m)}+b) ]$

计算过程$w^T X + b$细化如下图

![png](/assets/in-post/deeplearning.ai-1/1-12.png)

---

### 11 向量化实现逻辑回归反向传播


回顾一下反向传播计算过程

首先，通过前向传播，我们得到了 $A=[a^{(1)},a^{(2)},...,a^{(m)}]$ , 标签 $Y=[y^{(1)},y^{(2)},...,y^{(m)}]$

这时，我们可以计算出所有样本的 $dz^{(i)}$ 并放入一个矩阵dZ中

$dZ=[dz^{(1)},dz^{(2)},...,dz^{(m)}]=[(a^{(1)}-y^{(1)}),(a^{(2)}-y^{(2)}),...,(a^{(m)}-y^{(m)})]=A-Y$
