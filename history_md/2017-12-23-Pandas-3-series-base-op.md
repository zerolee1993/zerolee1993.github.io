---
layout:     post
title:      "Pandas Series序列基本操作"
subtitle:   "Python数据分析库Pandas简单使用（3/3）"
date:       2017-12-23
author:     "Zero"
#cover: "/assets/in-post/python1/bg.jpg"
categories: technology
tags: Pandas
---


### 简介

本文针对Pandas中的另一个核心数据结构Series进行了介绍，并列举了可执行代码，这里只是冰山一角，序列操作的实用性很强，列如在处理时间序列数据时
比如我们要进行一个股票的预测或降水量预测等，都是时间序列数据，我们拿到的原始数据一定不会太完美，pandas会很好的将时间序列数据进行处理后作为训练的输入
本文仅仅对Series最简单的部分做了介绍，之后会有时间序列以及股票预测的相关博客进行进一步深入的应用

---

### 目录

* 纲要
{:toc}

---

#### 1、数据结构Series

- DataFrame是pandas中的核心数据结构，Series可以理解成DataFrame的一行或一列
- 通过取得某一列得到Series数据，Series默认自动生成自然数索引
- Series支持切片方式取值
- 可以使用Series的value属性获得对应的包含数据的ndarray
- 使用文件：[fandango_score_comparison.csv](/assets/in-post/pandas3/fandango_score_comparison.csv)

```python
import pandas as pd
from pandas import Series

# 加载电影评分数据文件
fandango = pd.read_csv('fandango_score_comparison.csv')
# 打印前三列表头
print (fandango.columns[:3])
```

    Index(['FILM', 'RottenTomatoes', 'RottenTomatoes_User'], dtype='object')



```python
# 得到某一列Series
series_film = fandango['FILM']
print (type(series_film))
# Series支持切片方式取值
print (series_film[0:3])
# 查看Series的values，返回的是一个一维的ndarray
film_names = series_film.values
print (film_names.shape)
```

    <class 'pandas.core.series.Series'>
    0    Avengers: Age of Ultron (2015)
    1                 Cinderella (2015)
    2                    Ant-Man (2015)
    Name: FILM, dtype: object
    (146,)

---

#### 2、自定义Series的索引

- 使用Series创建序列数据，通过index可选参数可指定索引


```python
# 获得一个以电影名为索引，RottenTomatoes评分为数据值的Series
rt_scores = fandango['RottenTomatoes'].values
series_custom = Series(rt_scores, index=film_names)
# 以film name作为索引取出对应电影的RottenTomatoes评分
scores1 = series_custom[['Cinderella (2015)', 'Ant-Man (2015)']]
# 自定义索引的Series依然支持切片
scores2 = series_custom[0:3]
print (scores1)
print (scores2)
```

    Cinderella (2015)    85
    Ant-Man (2015)       80
    dtype: int64
    Avengers: Age of Ultron (2015)    74
    Cinderella (2015)                 85
    Ant-Man (2015)                    80
    dtype: int64

---

#### 3、数据排序

- 使用sort_index根据索引排序
- 使用sort_values根据值排序


```python
sort_index_data = series_custom.sort_index()
print (sort_index_data[0:5])
sort_values_data = series_custom.sort_values()
print (sort_values_data[0:5])
```

    '71 (2015)                    97
    5 Flights Up (2015)           52
    A Little Chaos (2015)         40
    A Most Violent Year (2014)    90
    About Elly (2015)             97
    dtype: int64
    Paul Blart: Mall Cop 2 (2015)    5
    Hitman: Agent 47 (2015)          7
    Hot Pursuit (2015)               8
    Fantastic Four (2015)            9
    Taken 3 (2015)                   9
    dtype: int64
