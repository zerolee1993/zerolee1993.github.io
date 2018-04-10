---
layout:     post
title:      "Pandas 基本操作"
subtitle:   "Python数据分析库Pandas简单使用（1/3）"
date:       2017-12-22
author:     "Zero"
#cover: "/assets/in-post/python1/bg.jpg"
categories: technology
tags: Pandas
---

### 简介

Pandas是Python的数据分析库，是基于Numpy的一种数据分析工具
对Numpy库进行了封装，帮助我们更加方便的进行数据分析
本文简述了Pandas的常用方法，并列举了可执行代码

---

### 目录

* 纲要
{:toc}

---

#### 1、读取csv文件
- pandas的核心数据结构DataFrame
- 使用read_csv读取csv文件
- 默认数据的第一行是列名
- csv是以逗号为分割的文件，可以用excel查看
- 使用文件：[food_info.csv](/assets/in-post/pandas1/food_info.csv)

```python
import pandas as pd

# 读取csv文件，这个文件记录了各种食物中各种物质的含量
food_info = pd.read_csv('data/food_info.csv')
print (type(food_info))
```

    <class 'pandas.core.frame.DataFrame'>

---

#### 2、查看数据的属性
- 使用dtypes属性可以查看所有列的数据类型，值得注意的是，String类型在pandas中是object
- 使用columns属性列出所有列名
- 使用shape属性查看数据规模


```python
# 打印前五列的数据类型
print (food_info.dtypes[0:5])
```

    NDB_No           int64
    Shrt_Desc       object
    Water_(g)      float64
    Energ_Kcal       int64
    Protein_(g)    float64
    dtype: object



```python
# 取出所有的列名
food_info.columns
```




    Index(['NDB_No', 'Shrt_Desc', 'Water_(g)', 'Energ_Kcal', 'Protein_(g)',
           'Lipid_Tot_(g)', 'Ash_(g)', 'Carbohydrt_(g)', 'Fiber_TD_(g)',
           'Sugar_Tot_(g)', 'Calcium_(mg)', 'Iron_(mg)', 'Magnesium_(mg)',
           'Phosphorus_(mg)', 'Potassium_(mg)', 'Sodium_(mg)', 'Zinc_(mg)',
           'Copper_(mg)', 'Manganese_(mg)', 'Selenium_(mcg)', 'Vit_C_(mg)',
           'Thiamin_(mg)', 'Riboflavin_(mg)', 'Niacin_(mg)', 'Vit_B6_(mg)',
           'Vit_B12_(mcg)', 'Vit_A_IU', 'Vit_A_RAE', 'Vit_E_(mg)', 'Vit_D_mcg',
           'Vit_D_IU', 'Vit_K_(mcg)', 'FA_Sat_(g)', 'FA_Mono_(g)', 'FA_Poly_(g)',
           'Cholestrl_(mg)'],
          dtype='object')




```python
# 查看数据规模，多少条数据，每条数据多少个指标
food_info.shape
```




    (8618, 36)

---

#### 3、按行取出数据
- 使用head函数查看前n条数据，默认5条
- 使用tail函数查看后n条数据，默认5条
- 使用loc[]通过索引定位数据


```python
# 查看前5条数据
food_info.head()
```




<div style='overflow-x:auto'>
<table border="1" class="dataframe" style='margin:0px'>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NDB_No</th>
      <th>Shrt_Desc</th>
      <th>Water_(g)</th>
      <th>Energ_Kcal</th>
      <th>Protein_(g)</th>
      <th>Lipid_Tot_(g)</th>
      <th>Ash_(g)</th>
      <th>Carbohydrt_(g)</th>
      <th>Fiber_TD_(g)</th>
      <th>Sugar_Tot_(g)</th>
      <th>...</th>
      <th>Vit_A_IU</th>
      <th>Vit_A_RAE</th>
      <th>Vit_E_(mg)</th>
      <th>Vit_D_mcg</th>
      <th>Vit_D_IU</th>
      <th>Vit_K_(mcg)</th>
      <th>FA_Sat_(g)</th>
      <th>FA_Mono_(g)</th>
      <th>FA_Poly_(g)</th>
      <th>Cholestrl_(mg)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>BUTTER WITH SALT</td>
      <td>15.87</td>
      <td>717</td>
      <td>0.85</td>
      <td>81.11</td>
      <td>2.11</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>0.06</td>
      <td>...</td>
      <td>2499.0</td>
      <td>684.0</td>
      <td>2.32</td>
      <td>1.5</td>
      <td>60.0</td>
      <td>7.0</td>
      <td>51.368</td>
      <td>21.021</td>
      <td>3.043</td>
      <td>215.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002</td>
      <td>BUTTER WHIPPED WITH SALT</td>
      <td>15.87</td>
      <td>717</td>
      <td>0.85</td>
      <td>81.11</td>
      <td>2.11</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>0.06</td>
      <td>...</td>
      <td>2499.0</td>
      <td>684.0</td>
      <td>2.32</td>
      <td>1.5</td>
      <td>60.0</td>
      <td>7.0</td>
      <td>50.489</td>
      <td>23.426</td>
      <td>3.012</td>
      <td>219.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1003</td>
      <td>BUTTER OIL ANHYDROUS</td>
      <td>0.24</td>
      <td>876</td>
      <td>0.28</td>
      <td>99.48</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>3069.0</td>
      <td>840.0</td>
      <td>2.80</td>
      <td>1.8</td>
      <td>73.0</td>
      <td>8.6</td>
      <td>61.924</td>
      <td>28.732</td>
      <td>3.694</td>
      <td>256.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>CHEESE BLUE</td>
      <td>42.41</td>
      <td>353</td>
      <td>21.40</td>
      <td>28.74</td>
      <td>5.11</td>
      <td>2.34</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>...</td>
      <td>721.0</td>
      <td>198.0</td>
      <td>0.25</td>
      <td>0.5</td>
      <td>21.0</td>
      <td>2.4</td>
      <td>18.669</td>
      <td>7.778</td>
      <td>0.800</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1005</td>
      <td>CHEESE BRICK</td>
      <td>41.11</td>
      <td>371</td>
      <td>23.24</td>
      <td>29.68</td>
      <td>3.18</td>
      <td>2.79</td>
      <td>0.0</td>
      <td>0.51</td>
      <td>...</td>
      <td>1080.0</td>
      <td>292.0</td>
      <td>0.26</td>
      <td>0.5</td>
      <td>22.0</td>
      <td>2.5</td>
      <td>18.764</td>
      <td>8.598</td>
      <td>0.784</td>
      <td>94.0</td>
    </tr>
  </tbody>
</table>
</div>
<p>5 rows × 36 columns</p>




```python
# 查看后2条数据
food_info.tail(2)
```




<div style='overflow-x:auto'>
<table border="1" class="dataframe" style='margin:0px'>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NDB_No</th>
      <th>Shrt_Desc</th>
      <th>Water_(g)</th>
      <th>Energ_Kcal</th>
      <th>Protein_(g)</th>
      <th>Lipid_Tot_(g)</th>
      <th>Ash_(g)</th>
      <th>Carbohydrt_(g)</th>
      <th>Fiber_TD_(g)</th>
      <th>Sugar_Tot_(g)</th>
      <th>...</th>
      <th>Vit_A_IU</th>
      <th>Vit_A_RAE</th>
      <th>Vit_E_(mg)</th>
      <th>Vit_D_mcg</th>
      <th>Vit_D_IU</th>
      <th>Vit_K_(mcg)</th>
      <th>FA_Sat_(g)</th>
      <th>FA_Mono_(g)</th>
      <th>FA_Poly_(g)</th>
      <th>Cholestrl_(mg)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8616</th>
      <td>90560</td>
      <td>SNAIL RAW</td>
      <td>79.2</td>
      <td>90</td>
      <td>16.1</td>
      <td>1.4</td>
      <td>1.3</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>100.0</td>
      <td>30.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.361</td>
      <td>0.259</td>
      <td>0.252</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>8617</th>
      <td>93600</td>
      <td>TURTLE GREEN RAW</td>
      <td>78.5</td>
      <td>89</td>
      <td>19.8</td>
      <td>0.5</td>
      <td>1.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>100.0</td>
      <td>30.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.127</td>
      <td>0.088</td>
      <td>0.170</td>
      <td>50.0</td>
    </tr>
  </tbody>
</table>
</div>
<p>2 rows × 36 columns</p>



```python
# 根据索引取出单条数据
food_info.loc[1]
```




    NDB_No                                 1002
    Shrt_Desc          BUTTER WHIPPED WITH SALT
    Water_(g)                             15.87
    Energ_Kcal                              717
    Protein_(g)                            0.85
    Lipid_Tot_(g)                         81.11
    Ash_(g)                                2.11
    Carbohydrt_(g)                         0.06
    Fiber_TD_(g)                              0
    Sugar_Tot_(g)                          0.06
    Calcium_(mg)                             24
    Iron_(mg)                              0.16
    Magnesium_(mg)                            2
    Phosphorus_(mg)                          23
    Potassium_(mg)                           26
    Sodium_(mg)                             659
    Zinc_(mg)                              0.05
    Copper_(mg)                           0.016
    Manganese_(mg)                        0.004
    Selenium_(mcg)                            1
    Vit_C_(mg)                                0
    Thiamin_(mg)                          0.005
    Riboflavin_(mg)                       0.034
    Niacin_(mg)                           0.042
    Vit_B6_(mg)                           0.003
    Vit_B12_(mcg)                          0.13
    Vit_A_IU                               2499
    Vit_A_RAE                               684
    Vit_E_(mg)                             2.32
    Vit_D_mcg                               1.5
    Vit_D_IU                                 60
    Vit_K_(mcg)                               7
    FA_Sat_(g)                           50.489
    FA_Mono_(g)                          23.426
    FA_Poly_(g)                           3.012
    Cholestrl_(mg)                          219
    Name: 1, dtype: object




```python
# 根据索引范围取数据
food_info.loc[2:4]
```




<div style='overflow-x:auto'>
<table border="1" class="dataframe" style='margin:0px'>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NDB_No</th>
      <th>Shrt_Desc</th>
      <th>Water_(g)</th>
      <th>Energ_Kcal</th>
      <th>Protein_(g)</th>
      <th>Lipid_Tot_(g)</th>
      <th>Ash_(g)</th>
      <th>Carbohydrt_(g)</th>
      <th>Fiber_TD_(g)</th>
      <th>Sugar_Tot_(g)</th>
      <th>...</th>
      <th>Vit_A_IU</th>
      <th>Vit_A_RAE</th>
      <th>Vit_E_(mg)</th>
      <th>Vit_D_mcg</th>
      <th>Vit_D_IU</th>
      <th>Vit_K_(mcg)</th>
      <th>FA_Sat_(g)</th>
      <th>FA_Mono_(g)</th>
      <th>FA_Poly_(g)</th>
      <th>Cholestrl_(mg)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1003</td>
      <td>BUTTER OIL ANHYDROUS</td>
      <td>0.24</td>
      <td>876</td>
      <td>0.28</td>
      <td>99.48</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>3069.0</td>
      <td>840.0</td>
      <td>2.80</td>
      <td>1.8</td>
      <td>73.0</td>
      <td>8.6</td>
      <td>61.924</td>
      <td>28.732</td>
      <td>3.694</td>
      <td>256.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>CHEESE BLUE</td>
      <td>42.41</td>
      <td>353</td>
      <td>21.40</td>
      <td>28.74</td>
      <td>5.11</td>
      <td>2.34</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>...</td>
      <td>721.0</td>
      <td>198.0</td>
      <td>0.25</td>
      <td>0.5</td>
      <td>21.0</td>
      <td>2.4</td>
      <td>18.669</td>
      <td>7.778</td>
      <td>0.800</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1005</td>
      <td>CHEESE BRICK</td>
      <td>41.11</td>
      <td>371</td>
      <td>23.24</td>
      <td>29.68</td>
      <td>3.18</td>
      <td>2.79</td>
      <td>0.0</td>
      <td>0.51</td>
      <td>...</td>
      <td>1080.0</td>
      <td>292.0</td>
      <td>0.26</td>
      <td>0.5</td>
      <td>22.0</td>
      <td>2.5</td>
      <td>18.764</td>
      <td>8.598</td>
      <td>0.784</td>
      <td>94.0</td>
    </tr>
  </tbody>
</table>
</div>
<p>3 rows × 36 columns</p>



```python
# 根据索引范围及取值跨度取数据
food_info.loc[0:10:3]
```




<div style='overflow-x:auto'>
<table border="1" class="dataframe" style='margin:0px'>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NDB_No</th>
      <th>Shrt_Desc</th>
      <th>Water_(g)</th>
      <th>Energ_Kcal</th>
      <th>Protein_(g)</th>
      <th>Lipid_Tot_(g)</th>
      <th>Ash_(g)</th>
      <th>Carbohydrt_(g)</th>
      <th>Fiber_TD_(g)</th>
      <th>Sugar_Tot_(g)</th>
      <th>...</th>
      <th>Vit_A_IU</th>
      <th>Vit_A_RAE</th>
      <th>Vit_E_(mg)</th>
      <th>Vit_D_mcg</th>
      <th>Vit_D_IU</th>
      <th>Vit_K_(mcg)</th>
      <th>FA_Sat_(g)</th>
      <th>FA_Mono_(g)</th>
      <th>FA_Poly_(g)</th>
      <th>Cholestrl_(mg)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>BUTTER WITH SALT</td>
      <td>15.87</td>
      <td>717</td>
      <td>0.85</td>
      <td>81.11</td>
      <td>2.11</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>0.06</td>
      <td>...</td>
      <td>2499.0</td>
      <td>684.0</td>
      <td>2.32</td>
      <td>1.5</td>
      <td>60.0</td>
      <td>7.0</td>
      <td>51.368</td>
      <td>21.021</td>
      <td>3.043</td>
      <td>215.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>CHEESE BLUE</td>
      <td>42.41</td>
      <td>353</td>
      <td>21.40</td>
      <td>28.74</td>
      <td>5.11</td>
      <td>2.34</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>...</td>
      <td>721.0</td>
      <td>198.0</td>
      <td>0.25</td>
      <td>0.5</td>
      <td>21.0</td>
      <td>2.4</td>
      <td>18.669</td>
      <td>7.778</td>
      <td>0.800</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1007</td>
      <td>CHEESE CAMEMBERT</td>
      <td>51.80</td>
      <td>300</td>
      <td>19.80</td>
      <td>24.26</td>
      <td>3.68</td>
      <td>0.46</td>
      <td>0.0</td>
      <td>0.46</td>
      <td>...</td>
      <td>820.0</td>
      <td>241.0</td>
      <td>0.21</td>
      <td>0.4</td>
      <td>18.0</td>
      <td>2.0</td>
      <td>15.259</td>
      <td>7.023</td>
      <td>0.724</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1010</td>
      <td>CHEESE CHESHIRE</td>
      <td>37.65</td>
      <td>387</td>
      <td>23.37</td>
      <td>30.60</td>
      <td>3.60</td>
      <td>4.78</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>985.0</td>
      <td>233.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.475</td>
      <td>8.671</td>
      <td>0.870</td>
      <td>103.0</td>
    </tr>
  </tbody>
</table>
</div>
<p>4 rows × 36 columns</p>

---

#### 4、按列取出数据
- 使用列名取出某列或某几列


```python
# 取出列名为‘NDB_No’的数据
ndb_col = food_info['NDB_No']
print (ndb_col.head())
```

    0    1001
    1    1002
    2    1003
    3    1004
    4    1005
    Name: NDB_No, dtype: int64



```python
# 取出列名为'Zinc_(mg)','Copper_(mg)'的数据
# 注意这里需要两层中括号，或者提前定义一个list变量
zinc_copper = food_info[['Zinc_(mg)','Copper_(mg)']]
print (zinc_copper.head())
```

       Zinc_(mg)  Copper_(mg)
    0       0.09        0.000
    1       0.05        0.016
    2       0.01        0.001
    3       2.66        0.040
    4       2.60        0.024



```python
# 取出列名以(g)为结尾的前三条数据
# 先转换列命为list
col_names = food_info.columns.tolist()
# 定义一个list存放以(g)结尾的列名
need_columns = []
for c in col_names:
    if c.endswith('(g)'):
        need_columns.append(c)
need_data = food_info[need_columns]
print (need_data.head(3))
```

       Water_(g)  Protein_(g)  Lipid_Tot_(g)  Ash_(g)  Carbohydrt_(g)  \
    0      15.87         0.85          81.11     2.11            0.06   
    1      15.87         0.85          81.11     2.11            0.06   
    2       0.24         0.28          99.48     0.00            0.00   

       Fiber_TD_(g)  Sugar_Tot_(g)  FA_Sat_(g)  FA_Mono_(g)  FA_Poly_(g)  
    0           0.0           0.06      51.368       21.021        3.043  
    1           0.0           0.06      50.489       23.426        3.012  
    2           0.0           0.00      61.924       28.732        3.694  

---

#### 5、数据的处理
- 将某列数据进行简单的运算处理
- 通过列数据之间的组合得到新的有意义的列
- 将某列数据归一化
- 按照某一列排序


```python
# 其中一列的mg转化为g
iron_g = food_info['Iron_(mg)'] / 1000
# 添加一列
food_info['Iron_(g)'] = iron_g
food_info.head()
```




<div style="overflow-x:auto">
<table border="1" class="dataframe" style='margin:0px'>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NDB_No</th>
      <th>Shrt_Desc</th>
      <th>Water_(g)</th>
      <th>Energ_Kcal</th>
      <th>Protein_(g)</th>
      <th>Lipid_Tot_(g)</th>
      <th>Ash_(g)</th>
      <th>Carbohydrt_(g)</th>
      <th>Fiber_TD_(g)</th>
      <th>Sugar_Tot_(g)</th>
      <th>...</th>
      <th>Vit_A_RAE</th>
      <th>Vit_E_(mg)</th>
      <th>Vit_D_mcg</th>
      <th>Vit_D_IU</th>
      <th>Vit_K_(mcg)</th>
      <th>FA_Sat_(g)</th>
      <th>FA_Mono_(g)</th>
      <th>FA_Poly_(g)</th>
      <th>Cholestrl_(mg)</th>
      <th>Iron_(g)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>BUTTER WITH SALT</td>
      <td>15.87</td>
      <td>717</td>
      <td>0.85</td>
      <td>81.11</td>
      <td>2.11</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>0.06</td>
      <td>...</td>
      <td>684.0</td>
      <td>2.32</td>
      <td>1.5</td>
      <td>60.0</td>
      <td>7.0</td>
      <td>51.368</td>
      <td>21.021</td>
      <td>3.043</td>
      <td>215.0</td>
      <td>0.00002</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002</td>
      <td>BUTTER WHIPPED WITH SALT</td>
      <td>15.87</td>
      <td>717</td>
      <td>0.85</td>
      <td>81.11</td>
      <td>2.11</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>0.06</td>
      <td>...</td>
      <td>684.0</td>
      <td>2.32</td>
      <td>1.5</td>
      <td>60.0</td>
      <td>7.0</td>
      <td>50.489</td>
      <td>23.426</td>
      <td>3.012</td>
      <td>219.0</td>
      <td>0.00016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1003</td>
      <td>BUTTER OIL ANHYDROUS</td>
      <td>0.24</td>
      <td>876</td>
      <td>0.28</td>
      <td>99.48</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>840.0</td>
      <td>2.80</td>
      <td>1.8</td>
      <td>73.0</td>
      <td>8.6</td>
      <td>61.924</td>
      <td>28.732</td>
      <td>3.694</td>
      <td>256.0</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>CHEESE BLUE</td>
      <td>42.41</td>
      <td>353</td>
      <td>21.40</td>
      <td>28.74</td>
      <td>5.11</td>
      <td>2.34</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>...</td>
      <td>198.0</td>
      <td>0.25</td>
      <td>0.5</td>
      <td>21.0</td>
      <td>2.4</td>
      <td>18.669</td>
      <td>7.778</td>
      <td>0.800</td>
      <td>75.0</td>
      <td>0.00031</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1005</td>
      <td>CHEESE BRICK</td>
      <td>41.11</td>
      <td>371</td>
      <td>23.24</td>
      <td>29.68</td>
      <td>3.18</td>
      <td>2.79</td>
      <td>0.0</td>
      <td>0.51</td>
      <td>...</td>
      <td>292.0</td>
      <td>0.26</td>
      <td>0.5</td>
      <td>22.0</td>
      <td>2.5</td>
      <td>18.764</td>
      <td>8.598</td>
      <td>0.784</td>
      <td>94.0</td>
      <td>0.00043</td>
    </tr>
  </tbody>
</table>
</div>
<p>5 rows × 37 columns</p>



```python
# 两列组合运算
water_energy = food_info['Water_(g)'] * food_info['Energ_Kcal']
food_info['water_energy'] = water_energy
food_info.head()
```



<div style='overflow-x:auto'>
<table border="1" class="dataframe" style='margin:0px'>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NDB_No</th>
      <th>Shrt_Desc</th>
      <th>Water_(g)</th>
      <th>Energ_Kcal</th>
      <th>Protein_(g)</th>
      <th>Lipid_Tot_(g)</th>
      <th>Ash_(g)</th>
      <th>Carbohydrt_(g)</th>
      <th>Fiber_TD_(g)</th>
      <th>Sugar_Tot_(g)</th>
      <th>...</th>
      <th>Vit_E_(mg)</th>
      <th>Vit_D_mcg</th>
      <th>Vit_D_IU</th>
      <th>Vit_K_(mcg)</th>
      <th>FA_Sat_(g)</th>
      <th>FA_Mono_(g)</th>
      <th>FA_Poly_(g)</th>
      <th>Cholestrl_(mg)</th>
      <th>Iron_(g)</th>
      <th>water_energy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>BUTTER WITH SALT</td>
      <td>15.87</td>
      <td>717</td>
      <td>0.85</td>
      <td>81.11</td>
      <td>2.11</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>0.06</td>
      <td>...</td>
      <td>2.32</td>
      <td>1.5</td>
      <td>60.0</td>
      <td>7.0</td>
      <td>51.368</td>
      <td>21.021</td>
      <td>3.043</td>
      <td>215.0</td>
      <td>0.00002</td>
      <td>11378.79</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002</td>
      <td>BUTTER WHIPPED WITH SALT</td>
      <td>15.87</td>
      <td>717</td>
      <td>0.85</td>
      <td>81.11</td>
      <td>2.11</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>0.06</td>
      <td>...</td>
      <td>2.32</td>
      <td>1.5</td>
      <td>60.0</td>
      <td>7.0</td>
      <td>50.489</td>
      <td>23.426</td>
      <td>3.012</td>
      <td>219.0</td>
      <td>0.00016</td>
      <td>11378.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1003</td>
      <td>BUTTER OIL ANHYDROUS</td>
      <td>0.24</td>
      <td>876</td>
      <td>0.28</td>
      <td>99.48</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>2.80</td>
      <td>1.8</td>
      <td>73.0</td>
      <td>8.6</td>
      <td>61.924</td>
      <td>28.732</td>
      <td>3.694</td>
      <td>256.0</td>
      <td>0.00000</td>
      <td>210.24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>CHEESE BLUE</td>
      <td>42.41</td>
      <td>353</td>
      <td>21.40</td>
      <td>28.74</td>
      <td>5.11</td>
      <td>2.34</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>...</td>
      <td>0.25</td>
      <td>0.5</td>
      <td>21.0</td>
      <td>2.4</td>
      <td>18.669</td>
      <td>7.778</td>
      <td>0.800</td>
      <td>75.0</td>
      <td>0.00031</td>
      <td>14970.73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1005</td>
      <td>CHEESE BRICK</td>
      <td>41.11</td>
      <td>371</td>
      <td>23.24</td>
      <td>29.68</td>
      <td>3.18</td>
      <td>2.79</td>
      <td>0.0</td>
      <td>0.51</td>
      <td>...</td>
      <td>0.26</td>
      <td>0.5</td>
      <td>22.0</td>
      <td>2.5</td>
      <td>18.764</td>
      <td>8.598</td>
      <td>0.784</td>
      <td>94.0</td>
      <td>0.00043</td>
      <td>15251.81</td>
    </tr>
  </tbody>
</table>
</div>
<p>5 rows × 38 columns</p>



```python
# 求最值
max_calories = food_info['Energ_Kcal'].max()
# 归一化操作
normalized_calories = food_info['Energ_Kcal'] / max_calories
food_info['normalized_calories'] = normalized_calories
food_info.head()
```




<div style='overflow-x:auto'>
<table border="1" class="dataframe" style='margin:0px'>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NDB_No</th>
      <th>Shrt_Desc</th>
      <th>Water_(g)</th>
      <th>Energ_Kcal</th>
      <th>Protein_(g)</th>
      <th>Lipid_Tot_(g)</th>
      <th>Ash_(g)</th>
      <th>Carbohydrt_(g)</th>
      <th>Fiber_TD_(g)</th>
      <th>Sugar_Tot_(g)</th>
      <th>...</th>
      <th>Vit_D_mcg</th>
      <th>Vit_D_IU</th>
      <th>Vit_K_(mcg)</th>
      <th>FA_Sat_(g)</th>
      <th>FA_Mono_(g)</th>
      <th>FA_Poly_(g)</th>
      <th>Cholestrl_(mg)</th>
      <th>Iron_(g)</th>
      <th>water_energy</th>
      <th>normalized_calories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>BUTTER WITH SALT</td>
      <td>15.87</td>
      <td>717</td>
      <td>0.85</td>
      <td>81.11</td>
      <td>2.11</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>0.06</td>
      <td>...</td>
      <td>1.5</td>
      <td>60.0</td>
      <td>7.0</td>
      <td>51.368</td>
      <td>21.021</td>
      <td>3.043</td>
      <td>215.0</td>
      <td>0.00002</td>
      <td>11378.79</td>
      <td>0.794900</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002</td>
      <td>BUTTER WHIPPED WITH SALT</td>
      <td>15.87</td>
      <td>717</td>
      <td>0.85</td>
      <td>81.11</td>
      <td>2.11</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>0.06</td>
      <td>...</td>
      <td>1.5</td>
      <td>60.0</td>
      <td>7.0</td>
      <td>50.489</td>
      <td>23.426</td>
      <td>3.012</td>
      <td>219.0</td>
      <td>0.00016</td>
      <td>11378.79</td>
      <td>0.794900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1003</td>
      <td>BUTTER OIL ANHYDROUS</td>
      <td>0.24</td>
      <td>876</td>
      <td>0.28</td>
      <td>99.48</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>1.8</td>
      <td>73.0</td>
      <td>8.6</td>
      <td>61.924</td>
      <td>28.732</td>
      <td>3.694</td>
      <td>256.0</td>
      <td>0.00000</td>
      <td>210.24</td>
      <td>0.971175</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>CHEESE BLUE</td>
      <td>42.41</td>
      <td>353</td>
      <td>21.40</td>
      <td>28.74</td>
      <td>5.11</td>
      <td>2.34</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>...</td>
      <td>0.5</td>
      <td>21.0</td>
      <td>2.4</td>
      <td>18.669</td>
      <td>7.778</td>
      <td>0.800</td>
      <td>75.0</td>
      <td>0.00031</td>
      <td>14970.73</td>
      <td>0.391353</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1005</td>
      <td>CHEESE BRICK</td>
      <td>41.11</td>
      <td>371</td>
      <td>23.24</td>
      <td>29.68</td>
      <td>3.18</td>
      <td>2.79</td>
      <td>0.0</td>
      <td>0.51</td>
      <td>...</td>
      <td>0.5</td>
      <td>22.0</td>
      <td>2.5</td>
      <td>18.764</td>
      <td>8.598</td>
      <td>0.784</td>
      <td>94.0</td>
      <td>0.00043</td>
      <td>15251.81</td>
      <td>0.411308</td>
    </tr>
  </tbody>
</table>
</div>
<p>5 rows × 39 columns</p>




```python
# 排序，默认从小到大，inplace表示是否原来基础上调整
food_info.sort_values('Water_(g)', inplace = True)
food_info.head()
```




<div style='overflow-x:auto'>
<table border="1" class="dataframe" style='margin:0px'>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NDB_No</th>
      <th>Shrt_Desc</th>
      <th>Water_(g)</th>
      <th>Energ_Kcal</th>
      <th>Protein_(g)</th>
      <th>Lipid_Tot_(g)</th>
      <th>Ash_(g)</th>
      <th>Carbohydrt_(g)</th>
      <th>Fiber_TD_(g)</th>
      <th>Sugar_Tot_(g)</th>
      <th>...</th>
      <th>Vit_D_mcg</th>
      <th>Vit_D_IU</th>
      <th>Vit_K_(mcg)</th>
      <th>FA_Sat_(g)</th>
      <th>FA_Mono_(g)</th>
      <th>FA_Poly_(g)</th>
      <th>Cholestrl_(mg)</th>
      <th>Iron_(g)</th>
      <th>water_energy</th>
      <th>normalized_calories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>676</th>
      <td>4544</td>
      <td>SHORTENING HOUSEHOLD LARD&amp;VEG OIL</td>
      <td>0.0</td>
      <td>900</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>21.5</td>
      <td>40.3</td>
      <td>44.4</td>
      <td>10.9</td>
      <td>56.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.997783</td>
    </tr>
    <tr>
      <th>664</th>
      <td>4520</td>
      <td>FAT MUTTON TALLOW</td>
      <td>0.0</td>
      <td>902</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.7</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>47.3</td>
      <td>40.6</td>
      <td>7.8</td>
      <td>102.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>665</th>
      <td>4528</td>
      <td>OIL WALNUT</td>
      <td>0.0</td>
      <td>884</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>9.1</td>
      <td>22.8</td>
      <td>63.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.980044</td>
    </tr>
    <tr>
      <th>666</th>
      <td>4529</td>
      <td>OIL ALMOND</td>
      <td>0.0</td>
      <td>884</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>8.2</td>
      <td>69.9</td>
      <td>17.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.980044</td>
    </tr>
    <tr>
      <th>667</th>
      <td>4530</td>
      <td>OIL APRICOT KERNEL</td>
      <td>0.0</td>
      <td>884</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.3</td>
      <td>60.0</td>
      <td>29.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.980044</td>
    </tr>
  </tbody>
</table>
</div>
<p>5 rows × 39 columns</p>



```python
# 通过指定可选参数ascending为False，指定从大到小排序,无论怎样排序，NaN缺失值总在最后
food_info.sort_values('Water_(g)', inplace = True, ascending=False)
food_info.head()
```




<div style='overflow-x:auto'>
<table border="1" class="dataframe" style='margin:0px'>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NDB_No</th>
      <th>Shrt_Desc</th>
      <th>Water_(g)</th>
      <th>Energ_Kcal</th>
      <th>Protein_(g)</th>
      <th>Lipid_Tot_(g)</th>
      <th>Ash_(g)</th>
      <th>Carbohydrt_(g)</th>
      <th>Fiber_TD_(g)</th>
      <th>Sugar_Tot_(g)</th>
      <th>...</th>
      <th>Vit_D_mcg</th>
      <th>Vit_D_IU</th>
      <th>Vit_K_(mcg)</th>
      <th>FA_Sat_(g)</th>
      <th>FA_Mono_(g)</th>
      <th>FA_Poly_(g)</th>
      <th>Cholestrl_(mg)</th>
      <th>Iron_(g)</th>
      <th>water_energy</th>
      <th>normalized_calories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4209</th>
      <td>14076</td>
      <td>BEVERAGES ICELANDIC GLACIAL NAT SPRING H2O</td>
      <td>100.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4376</th>
      <td>14437</td>
      <td>WATER BTLD NON-CARBONATED CALISTOGA</td>
      <td>100.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4348</th>
      <td>14385</td>
      <td>WATER BTLD POLAND SPRING</td>
      <td>100.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00001</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4377</th>
      <td>14438</td>
      <td>WATER BTLD NON-CARBONATED CRYSTAL GEYSER</td>
      <td>100.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4378</th>
      <td>14439</td>
      <td>WATER BTLD NON-CARBONATED NAYA</td>
      <td>100.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>
<p>5 rows × 39 columns</p>
