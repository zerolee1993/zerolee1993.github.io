---
layout:     post
title:      "Seaborn画图风格设定"
subtitle:   "Python可视化库Seaborn简单使用（1/2）"
date:       2017-12-28
author:     "Zero"
#cover: "/assets/in-post/seaborn1/bg.jpg"
categories: technology
tags: Seaborn
---

### 简介

Seaborn在matplotlib的基础上进行了更高级的封装
我们能够使用Seaborn更美观、直观的展现数据
本文简述了如何使用Seaborn设定各种画图风格
在学会风格的设定后，你可以基于一种你最喜欢的风格，进行《Seaborn数据展示》

---

### 目录

* 纲要
{:toc}

---

#### 1、在这里，我们先来构造一个画图函数，方便之后画图时候的调用，并对比图像风格

```python
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 这行代码使之后画图完成后，执行代码直接展示图的效果
%matplotlib inline

# 构造画三角函数图的函数sinplot [plot:绘图]
def sinplot(flip=1):
    # 构造数据：在[1, 14]范围内平均取100个数字
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        # i * 0.5 表示横向偏移量，左加右减，这里是向左平移
        # 7 - i 表示幅度，即峰值，为1就是默认，越大，图像峰值越小
        # filp指将整体的值纵向拉长的倍数
        plt.plot(x, np.sin(x + i * 0.5) * (7 - i) * flip)

# 调用函数画图（这里使用的plt的风格）
sinplot()
```


![png](/assets/in-post/seaborn1/output_1_0.png)

---

#### 2、让我们试一试seaborn提供的风格
- 使用set()方法启动seaborn默认风格配置替换plt原始风格
- 使用set_style('dark')方法使用下面提到的主题风格
- Seaborn提供了五种主题风格：
  - darkgrid 黑色网格 [grid:网格]
  - whitegrid 白色网格
  - dark 黑色
  - white 白色
  - ticks 十字叉



```python
# 使用set方法更换plt风格为seaborn风格，即启动sns的默认配置
# 默认看上去应该是darkgrid
sns.set()
sinplot()
```


![png](/assets/in-post/seaborn1/output_3_0.png)



```python
# 使用set_style设置之前提到的五种风格之一
sns.set_style('whitegrid')
sinplot()
```


![png](/assets/in-post/seaborn1/output_4_0.png)



```python
# 使用盒图看一下效果，盒图语法先不用关心
sns.set_style('whitegrid')

# 构造一组正态分布的数据，shape为(20, 6)，然后加上向量 (0, 1, 2, 3, 4, 5)
# 其实是正态分布数据的每一行6个数，与向量中的6个数，按位置相加
data = np.random.normal(size=(20, 6)) + np.arange(6) / 2

# 使用boxplot画出盒图
sns.boxplot(data = data)
# 盒图中的whitegrid风格不再是网格而是条格
```



![png](/assets/in-post/seaborn1/output_5_1.png)



```python
# 尝试dark风格
sns.set_style('dark')
sinplot()
```


![png](/assets/in-post/seaborn1/output_6_0.png)



```python
# 尝试white风格
sns.set_style('white')
sinplot()
```


![png](/assets/in-post/seaborn1/output_7_0.png)



```python
# 尝试ticks风格
sns.set_style('ticks')
sinplot()
```


![png](/assets/in-post/seaborn1/output_8_0.png)

---

#### 3、尝试完这些默认风格之后，我们可以使用一些方法，对一些细节进行调整
- 使用despine()去掉上方和右方的刻度线
- 使用despine(offset=x)指定到图像轴线的距离
- 想画出不同风格的子图，可以使用with sns.axes_style('dark')开启一种画图风格，与其他代码画图风格分离


```python
# ticks风格去掉上面和右面的刻度线,注意了，要在画图方法后执行才生效
sns.set_style('ticks')
sinplot()
sns.despine()
```


![png](/assets/in-post/seaborn1/output_10_0.png)



```python
# 使用despine(offset=x)指定到图像轴线的距离
sns.set_style('whitegrid')
sinplot()
sns.despine(offset=20)
```


![png](/assets/in-post/seaborn1/output_11_0.png)



```python
# 有时候我们想画出不同风格的子图
# 使用with打开一种画图风格，with内部使用axes_style设置的风格
# with外部使用原来的风格，互不影响
sns.set_style('whitegrid')
with sns.axes_style('dark'):
    plt.subplot(211)
    sinplot()
plt.subplot(212)
sinplot(-1)
```


![png](/assets/in-post/seaborn1/output_12_0.png)

---

#### 4、使用set_context('paper')指定seaborn提供的布局
- 提供的布局选项：paper, talk, poster, notebook
- set_context中可以传入可选参数指定字体大小及线宽
- sns.set_context("notebook", font_scale=2, rc={'lines.linewidth': 3})




```python
sns.set_style('whitegrid')
# 指定布局为paper
sns.set_context("paper")
# 指定区域大小
plt.figure(figsize=(6, 4))
sinplot()
```


![png](/assets/in-post/seaborn1/output_14_0.png)



```python
sns.set_style('whitegrid')
# 指定布局为talk
sns.set_context("talk")
# 指定区域大小
plt.figure(figsize=(6, 4))
sinplot()
```


![png](/assets/in-post/seaborn1/output_15_0.png)



```python
sns.set_style('whitegrid')
# 指定布局为poster
sns.set_context("poster")
# 指定区域大小
plt.figure(figsize=(6, 4))
sinplot()
```


![png](/assets/in-post/seaborn1/output_16_0.png)



```python
sns.set_style('whitegrid')
# 指定布局为notebook
sns.set_context("notebook")
# 指定区域大小
plt.figure(figsize=(6, 4))
sinplot()
```


![png](/assets/in-post/seaborn1/output_17_0.png)



```python
# 使用set_context的可选参数指定字体大小及线宽
sns.set_style('whitegrid')
sns.set_context("notebook", font_scale=2, rc={'lines.linewidth': 3})
sinplot()
```


![png](/assets/in-post/seaborn1/output_18_0.png)

---

#### 5、现在我们开始关注颜色的配置和使用
- 如果我们不做配置，将使用默认调色板,默认调色板是6个循环的颜色
- 使用color_palette()传入颜色参数获得调色板,不传参数为默认色板
- 使用palplot()可以将调色板展示出来
- 在画图时，指定一个可选参数palette，即可使用创建的调色板
- set_palette()设置所有图的颜色
- 我们还可以使用xkcd颜色
- 渐变颜色和线性颜色的使用等


```python
# 查看sns的默认色板
# 在上面代码所有没有指定色板的画图中，就是使用的这个默认颜色主题
current_palette = sns.color_palette()
# palplot函数将颜色调色板中的值绘制成一个水平数组
sns.palplot(current_palette)
```


![png](/assets/in-post/seaborn1/output_20_0.png)



```python
# 当我们要用到6个以上颜色时，seaborn提供了一个很好的方案
# 在一个圆形颜色空间中找出均匀间隔的颜色
# 最常用方法是使用hls的颜色空间，这是RGB的一个简单转换
sns.palplot(sns.color_palette('hls', 8))
# 我们也能使用下面的方法，创建hls颜色空间的8个均值
# 可以使用可选参数指定亮度l和饱和度s的值，范围[0,1]
sns.palplot(sns.hls_palette(8, l=0.1, s=0.8))
```


![png](/assets/in-post/seaborn1/output_21_0.png)



![png](/assets/in-post/seaborn1/output_21_1.png)



```python
# 当进行成对对比时，可以使用另外一个颜色空间Paired
# 可以看到，得到的调色板，以一深一浅的方式成对出现
sns.palplot(sns.color_palette('Paired', 8))
```


![png](/assets/in-post/seaborn1/output_22_0.png)



```python
# 让我们来使用一个八个颜色的调色板，通过指定可选参数palette
# 为了看起来只管，我们用之前用过的一个盒图的例子，稍作修改
data = np.random.normal(size=(20, 8)) + np.arange(8) / 2
sns.boxplot(data=data, palette=sns.color_palette('hls', 8))
```



![png](/assets/in-post/seaborn1/output_23_1.png)



```python
# 通过调用xkcd_rgb字典用的颜色命名设置颜色,lw表示线宽
plt.plot([0, 1], [0, 1], sns.xkcd_rgb['pale red'], lw=3)
plt.plot([0, 1], [0, 2], sns.xkcd_rgb['medium green'], lw=3)
plt.plot([0, 1], [0, 3], sns.xkcd_rgb['denim blue'], lw=3)
```




    [<matplotlib.lines.Line2D at 0x277c87c5358>]




![png](/assets/in-post/seaborn1/output_24_1.png)



```python
# 使用xkcd_palette函数，获得xkcd指定的调色板
sns.palplot(sns.xkcd_palette(['red', 'blue', 'orange']))
```


![png](/assets/in-post/seaborn1/output_25_0.png)



```python
# 连续渐变色板
sns.palplot(sns.color_palette('Blues'))
sns.palplot(sns.color_palette('Blues_r'))
sns.palplot(sns.color_palette('BuGn', 8))
sns.palplot(sns.color_palette('BuGn_r',10))

# 也可使用light_palette/dark_palette调用渐变色板
# 由浅入深
sns.palplot(sns.light_palette('green'))
# 通过reverse=True，将颜色顺序翻转
sns.palplot(sns.light_palette('green', reverse=True))
# 由深入浅
sns.palplot(sns.dark_palette('purple', 8))
```


![png](/assets/in-post/seaborn1/output_26_0.png)



![png](/assets/in-post/seaborn1/output_26_1.png)



![png](/assets/in-post/seaborn1/output_26_2.png)



![png](/assets/in-post/seaborn1/output_26_3.png)



![png](/assets/in-post/seaborn1/output_26_4.png)



![png](/assets/in-post/seaborn1/output_26_5.png)



![png](/assets/in-post/seaborn1/output_26_6.png)



```python
# 使用渐变色板画一个图
x, y = np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=300).T
pal = sns.dark_palette('green', as_cmap=True)
sns.kdeplot(x, y, cmap=pal)
```



![png](/assets/in-post/seaborn1/output_27_1.png)



```python
# 可以使用cubehelix色调线性变换方式
sns.palplot(sns.color_palette('cubehelix', 8))
# cubehelix_palette同样可以使用线性调色板，可以通过start、rot指定颜色区间
sns.palplot(sns.cubehelix_palette(8, start=.5, rot=-.75))
```


![png](/assets/in-post/seaborn1/output_28_0.png)



![png](/assets/in-post/seaborn1/output_28_1.png)
