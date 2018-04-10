---
layout:     post
title:      "Matplotlib绘图"
subtitle:   "Python可视化库Matplotlib简单使用（1/1）"
date:       2017-12-28
author:     "Zero"
#cover: "/assets/in-post/seaborn1/bg.jpg"
categories: technology
tags: Matplotlib
---

### 简介

matplotlib是python的作图工具库，使用matplot可以将数据制作成各种图表

---

### 目录

* 纲要
{:toc}

---

#### 1.折线图


```python
# 查看用于作图的数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 自动作图，省略plt.show()操作
%matplotlib inline

# 读取美国每年失业率文件
unrate = pd.read_csv('unrate.csv')
# 转换日志列的格式
unrate['DATE'] = pd.to_datetime(unrate['DATE'])
# 查看前12个月的数据
print (unrate.head(12))
```

             DATE  VALUE
    0  1948-01-01    3.4
    1  1948-02-01    3.8
    2  1948-03-01    4.0
    3  1948-04-01    3.9
    4  1948-05-01    3.5
    5  1948-06-01    3.6
    6  1948-07-01    3.6
    7  1948-08-01    3.9
    8  1948-09-01    3.8
    9  1948-10-01    3.7
    10 1948-11-01    3.8
    11 1948-12-01    4.0



```python
# 画图操作，不传入参数，表示不画东西
plt.plot()
```



![png](/assets/in-post/matplotlib1/output_2_1.png)



```python
# 取前12条数据作图
first_twelve = unrate[0:12]
plt.plot(first_twelve['DATE'], first_twelve['VALUE'])
```




    [<matplotlib.lines.Line2D at 0xa753d30>]




![png](/assets/in-post/matplotlib1/output_3_1.png)



```python
# 让X轴更美观
plt.plot(first_twelve['DATE'], first_twelve['VALUE'])
# 指定角度
plt.xticks(rotation=45)
```




    (array([711158., 711218., 711279., 711340., 711401., 711462.]),
     <a list of 6 Text xticklabel objects>)




![png](/assets/in-post/matplotlib1/output_4_1.png)



```python
# 增加标题、X轴Y轴说明
plt.plot(first_twelve['DATE'], first_twelve['VALUE'])
plt.xticks(rotation=90)
plt.xlabel('Month')
plt.ylabel('Unemlployment Rate')
plt.title('MonthLy Unemployment Trends, 1948')
plt.show()
```


![png](/assets/in-post/matplotlib1/output_5_0.png)



```python
# 在同一个图中展示两年的数据
unrate['MONTH'] = unrate['DATE'].dt.month
fig = plt.figure(figsize=(6, 3))
plt.plot(unrate[0:12]['MONTH'], unrate[0:12]['VALUE'], c='red')
plt.plot(unrate[12:24]['MONTH'], unrate[12:24]['VALUE'], c='green')
plt.show()
```


![png](/assets/in-post/matplotlib1/output_6_0.png)



```python
# 打印五年的折线图，并给每条折线打印标签（使用label，以及legend函数）
colors = ['red', 'green', 'blue', 'pink', 'orange']
for i in range(5):
    start_index = i*12
    end_index = (i+1)*12
    subset = unrate[start_index:end_index]
    label = str(1948+i)
    plt.plot(subset['MONTH'], subset['VALUE'], c=colors[i], label=label)
plt.legend(loc='best')
plt.xlabel('Month')
plt.ylabel('Unemlployment Rate')
plt.title('MonthLy Unemployment Trends, 1948')
plt.show()

```


![png](/assets/in-post/matplotlib1/output_7_0.png)


---

#### 2. 子图


```python
# 获得画图区间
fig = plt.figure()
# 创建子图中的第1个模块，前两个参数指定行列数
ax1 = fig.add_subplot(2, 2, 1)
# 创建子图中的第2、4个模块
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 4)
plt.show()
```


![png](/assets/in-post/matplotlib1/output_9_0.png)



```python
# 通过figsize指定区间的长宽
fig = plt.figure(figsize=(5,6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)

ax1.plot(np.random.randint(1, 5, 5), np.arange(5))
ax2.plot(np.arange(10), np.sin(np.arange(10)))
plt.show()
```


![png](/assets/in-post/matplotlib1/output_10_0.png)

---

#### 3. 条形图(bar图)


```python
# 加载电影评分的实例文件
reviews = pd.read_csv('fandango_score_comparison.csv')
# 指定一些列(电影名和五个媒体的评分)，并取出数据
cols = ['FILM', 'RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']
norm_reviews = reviews[cols]
#print (norm_reviews[:1])
# 指定评分列
num_cols = ['RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']
# 每个柱子的高度
bar_heights = norm_reviews.loc[0, num_cols].values
# 每个柱子到X轴的距离
bar_positions = np.arange(5) + 0.75

tick_positions = range(1,6)
# 声明区间和子图
fig, ax = plt.subplots()
ax.bar(bar_positions, bar_heights, 0.3) # 创建bar图 0.3表示bar的宽度
# 使用barh函数可画出横的bar图，参数不变，但是后面的label注释等需要随之调整
ax.set_xticks(tick_positions)
ax.set_xticklabels(num_cols, rotation=45)
ax.set_xlabel('Rating Source')
ax.set_ylabel('Average Rating')
ax.set_title('Average User Rating For Avengers: Age of Ultron (2015)')
```




    Text(0.5,1,'Average User Rating For Avengers: Age of Ultron (2015)')




![png](/assets/in-post/matplotlib1/output_12_1.png)

---

#### 4. 散点图


```python
# 画散点图
fig, ax = plt.subplots()
ax.scatter(norm_reviews['Fandango_Ratingvalue'], norm_reviews['RT_user_norm'])
ax.set_xlabel('Fandango')
ax.set_ylabel('Rotten Tomatoes')
plt.show()
```


![png](/assets/in-post/matplotlib1/output_14_0.png)

---

#### 5. 柱状图


```python
# 加载电影评分的实例文件
reviews = pd.read_csv('fandango_score_comparison.csv')
# 指定一些列(电影名和五个媒体的评分)，并取出数据
cols = ['FILM', 'RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']
norm_reviews = reviews[cols]
fig, ax = plt.subplots()
# 不指定bins，会默认指定bins位10
# ax.hist(norm_reviews['Fandango_Ratingvalue'])
# 指定bin为20
# ax.hist(norm_reviews['Fandango_Ratingvalue'],bins=20)
# 指定range，只显示范围内数据
ax.hist(norm_reviews['Fandango_Ratingvalue'], range=(4, 5),bins=20)

ax.set_ylim(0,20)#指定y轴区间
plt.show()
```


![png](/assets/in-post/matplotlib1/output_16_0.png)

---

#### 6. 盒图


```python
# 加载电影评分的实例文件
reviews = pd.read_csv('fandango_score_comparison.csv')
# 指定一些列(电影名和五个媒体的评分)，并取出数据
cols = ['FILM', 'RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']
norm_reviews = reviews[cols]
fig, ax = plt.subplots()
ax.boxplot(norm_reviews['RT_user_norm']) #盒图展示了1/4 1/2 3/4数据的位置
ax.set_xticklabels(['Rotten Tomatoes'])
ax.set_ylim(0, 5)
plt.show()
```


![png](/assets/in-post/matplotlib1/output_18_0.png)



```python
# 绘制多个盒图
num_cols = ['RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue']
fig, ax = plt.subplots()
ax.boxplot(norm_reviews[num_cols].values)
ax.set_xticklabels(num_cols, rotation=45)
ax.set_ylim(0,5)
plt.show()
```


![png](/assets/in-post/matplotlib1/output_19_0.png)
