---
layout:     post
title:      "Pandas-2-数据预处理实例"
subtitle:   "Python数据分析库Pandas学习笔记（2/3）"
date:       2017-12-22
author:     "Zero"
#cover: "/assets/in-post/python1/bg.jpg"
categories: technology
tags: Pandas
---

### 简介

本文结合《Pandas基本操作》相关介绍，针对于泰坦尼克号获救数据
进行了一些预处理操作，对pandas操作数据进行巩固

---

### 目录

* 纲要
{:toc}

---

#### 1、加载数据

- read_csv函数读取泰坦尼克号获救数据
- 使用文件：[titanic_train.vsv](/assets/in-post/pandas2/titanic_train.csv)


```python
import pandas as pd
import numpy as np

# 加载泰坦尼克号数据文件
titanic_survival = pd.read_csv('titanic_train.csv')
titanic_survival.head(10)
```




<div style='overflow-x:auto'>
<table border="1" class="dataframe" style='margin:0px'>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



#### 2、NaN空值数据处理

- 通过pd.isnull函数获得值为True/False的数据，这份数据可以作为取值的索引，进而过滤数据


```python
# 取出年龄列数据
age = titanic_survival['Age']
print (age.head(3))
# 判断年龄是否为空,得到的为与age同结构的数据，数据值为True/False
age_is_null = pd.isnull(age)
print (age_is_null.head(3))
# 将age_is_null作为索引,取出所有age是空的数据
age_null = age[age_is_null]
print (age_null.head(3))
# 获得age为空的数据的数量
age_null_count = len(age_null)
print (age_null_count)
# 获得年龄不为空的数据
age_not_null = age[age_is_null == False]
print (age_not_null.head(3))
```

    0    22.0
    1    38.0
    2    26.0
    Name: Age, dtype: float64
    0    False
    1    False
    2    False
    Name: Age, dtype: bool
    5    NaN
    17   NaN
    19   NaN
    Name: Age, dtype: float64
    177
    0    22.0
    1    38.0
    2    26.0
    Name: Age, dtype: float64


- 使用dropna函数放弃包含NaN的数据
- 使用fillna函数填充缺失值(代码略)


```python
# 可选参数axis指定为1，放弃包含NaN数据的列
drop_na_columns = titanic_survival.dropna(axis=1)
# 可选参数axis指定为0，放弃包含NaN数据的行
# 在这里指定subset，放弃subset中指定的列中包含NaN的数据行
new_titanic_survival = titanic_survival.dropna(axis=0, subset=['Age','Sex'])
new_titanic_survival.head(10)
```




<div style='overflow-x:auto'>
<table border="1" class="dataframe" style='margin:0px'>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>G6</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



#### 3、求某列平均值


```python
# 求平均年龄，没有对数据处理时候，数据中有缺失值，会导致均值计算无效
mean_age = sum(titanic_survival['Age']) / len(titanic_survival['Age'])
print (mean_age)
# 获得年龄不为空的数据
age_not_null = age[pd.isnull(age) == False]
correct_mean_age = sum(age_not_null) / len(age_not_null)
print (correct_mean_age)
# 像如上忽略NaN数据的方式，pandas中已经定义了对应的函数
print (titanic_survival['Age'].mean())
# 但是这种忽略缺失值的方式并不常用，通常会用均值或中值等对缺失值进行填充
```

    nan
    29.6991176471
    29.69911764705882


#### 4、分组统计

- 数据中的Pclass列代表船舱类别，取值为1或2或3
- 数据中的Fare列代表船舱价格
- 自己编码统计三类船舱的平均价格
- 数据中的Survived代表是否存活
- 使用pandas提供的分组统计函数pivot_table统计各船舱存活人数、存活率


```python
passenger_classes = [1, 2, 3]
fares_by_class = {}
for this_class in passenger_classes:
    pclass_rows = titanic_survival[titanic_survival["Pclass"] == this_class]
    pclass_fares = pclass_rows['Fare']
    fare_for_class = pclass_fares.mean()
    fares_by_class[this_class] = fare_for_class
print (fares_by_class)
```

    {1: 84.15468749999992, 2: 20.66218315217391, 3: 13.675550101832997}



```python
# 用pivot_table做类似如上的一个统计，统计三种船舱各自存活率
# aggfunc可选参数可指定具体统计操作，如求和、求均值
passenger_survival_num = titanic_survival.pivot_table(index='Pclass', values='Survived', aggfunc=np.sum)
passenger_survival_rate = titanic_survival.pivot_table(index='Pclass', values='Survived', aggfunc=np.mean)
print (passenger_survival_num)
print (passenger_survival_rate)
```

    Pclass
    1    136
    2     87
    3    119
    Name: Survived, dtype: int64
    Pclass
    1    0.629630
    2    0.472826
    3    0.242363
    Name: Survived, dtype: float64


#### 5、数据排序

- 使用sort_values指定列名进行排序
- ascending可选参数指定正序or倒序，默认为True，正序
- 排序后的索引值也打乱了，通过reset_index函数重置索引


```python
# 根据年龄排序
new_titanic_survival = titanic_survival.sort_values('Age', ascending=False)
new_titanic_survival.head(3)
```




<div style='overflow-x:auto'>
<table border="1" class="dataframe" style='margin:0px'>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>630</th>
      <td>631</td>
      <td>1</td>
      <td>1</td>
      <td>Barkworth, Mr. Algernon Henry Wilson</td>
      <td>male</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>27042</td>
      <td>30.0000</td>
      <td>A23</td>
      <td>S</td>
    </tr>
    <tr>
      <th>851</th>
      <td>852</td>
      <td>0</td>
      <td>3</td>
      <td>Svensson, Mr. Johan</td>
      <td>male</td>
      <td>74.0</td>
      <td>0</td>
      <td>0</td>
      <td>347060</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>493</th>
      <td>494</td>
      <td>0</td>
      <td>1</td>
      <td>Artagaveytia, Mr. Ramon</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17609</td>
      <td>49.5042</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 此时发现新排序的数据index是原来的，如果向更新index，使用如下操作
titanic_reindexed = new_titanic_survival.reset_index(drop=True)
titanic_reindexed.head(3)
```




<div style='overflow-x:auto'>
<table border="1" class="dataframe" style='margin:0px'>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>631</td>
      <td>1</td>
      <td>1</td>
      <td>Barkworth, Mr. Algernon Henry Wilson</td>
      <td>male</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>27042</td>
      <td>30.0000</td>
      <td>A23</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>852</td>
      <td>0</td>
      <td>3</td>
      <td>Svensson, Mr. Johan</td>
      <td>male</td>
      <td>74.0</td>
      <td>0</td>
      <td>0</td>
      <td>347060</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>494</td>
      <td>0</td>
      <td>1</td>
      <td>Artagaveytia, Mr. Ramon</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17609</td>
      <td>49.5042</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



#### 6、自定义函数

- 使用apply调用自定义函数
- 示例1：返回第100行数据
- 示例2：返回每列缺失值个数
- 示例3：使用自定义函数将年龄离散化
- 示例4：使用自定义函数将年龄转化为label
- 示例5：利用示例4的label数据，结合之前的pivot_table统计成年人和未成年人各自存活率


```python
# 返回第100行数据
def hundredth_row(column):
    hundredth_item = column.loc[99]
    return hundredth_item
hundredth_row = titanic_survival.apply(hundredth_row)
print (hundredth_row)
```

    PassengerId                  100
    Survived                       0
    Pclass                         2
    Name           Kantor, Mr. Sinai
    Sex                         male
    Age                           34
    SibSp                          1
    Parch                          0
    Ticket                    244367
    Fare                          26
    Cabin                        NaN
    Embarked                       S
    dtype: object



```python
# 返回每列缺失值个数
def null_count(column):
    column_null = pd.isnull(column)
    null = column[column_null]
    return len(null)

column_null_count = titanic_survival.apply(null_count)
print (column_null_count)
```

    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64



```python
# 使用自定义函数将年龄离散化
def is_minor(row):
    if row['Age'] < 18:
        return True
    else:
        return False
minors = titanic_survival.apply(is_minor, axis=1)
print (minors.head())
```

    0    False
    1    False
    2    False
    3    False
    4    False
    dtype: bool



```python
# 使用自定义函数将年龄转化为label
def generate_age_label(row):
    age = row['Age']
    if pd.isnull(age):
        return 'unknow'
    elif age < 18:
        return 'minor'
    else:
        return 'adult'
age_labels = titanic_survival.apply(generate_age_label, axis=1)
print (age_labels.head())
```

    0    adult
    1    adult
    2    adult
    3    adult
    4    adult
    dtype: object



```python
# 计算成年人、未成年人格子获救率
titanic_survival['age_label'] = age_labels
age_group_survival = titanic_survival.pivot_table(index='age_label', values='Survived')
print (age_group_survival)
```

    age_label
    adult     0.381032
    minor     0.539823
    unknow    0.293785
    Name: Survived, dtype: float64
