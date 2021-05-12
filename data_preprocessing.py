# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:15:46 2021

@author: 14373
"""


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")


train.columns

train.drop("Id", axis=1, inplace=True)
 
train["SalePrice"].describe()


sns.distplot(train["SalePrice"])

train["SalePrice"]=np.log(train["SalePrice"])

sns.distplot(train["SalePrice"])

train["SalePrice"].describe()

total_train=train.isnull().sum().reset_index(drop=True)
null_percentage=(train.isnull().sum()/train.isnull().count()*100).reset_index(drop=True)
columns=pd.Series(train.columns).reset_index(drop=True)

percentage_df=pd.DataFrame([columns,total_train,null_percentage]).T

percentage_df.columns=["name","total_train","null_percentage"]

percentage_df=percentage_df.sort_values(by="null_percentage", ascending=False)



###### 

# zero_null_columns=percentage_df.groupby("name")[percentage_df["null_percentage"]>0]

zero_null_columns=percentage_df[percentage_df["null_percentage"]==0]

zero_column_list=zero_null_columns["name"].to_list()

train_original=train.copy(deep=True)

train.drop(columns=zero_column_list,inplace=True)


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_df=train.select_dtypes(include=numerics)
categorical_df=train.select_dtypes(exclude=numerics)



# check=train_original[['BsmtHalfBath', 'BsmtFullBath', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1','MasVnrArea','GarageYrBlt', 'GarageArea', 'GarageCars']].isnull().sum()


####  LotFrontage is ~18% empty. we can use mean or median value to fill the NaN values based on Neighborhood or corresponding unique element
train_original["LotFrontage"]=train_original.groupby("Neighborhood").transform(lambda x:x.median())

train_original["LotFrontage"].isnull().sum()

train_original["GarageYrBlt"]=train_original["GarageYrBlt"].fillna(train_original["GarageYrBlt"].mode()[0])

train_original["GarageYrBlt"].isnull().sum()


numerical_df.drop("GarageYrBlt",axis=1, inplace=True)

numerical_list=numerical_df.columns.to_list()
categorical_list=categorical_df.columns.to_list()

for column in numerical_list:
    train_original[column]=train_original[column].fillna(0)
    
for column in categorical_list:
    train_original[column]=train_original[column].fillna(train_original[column].mode()[0])
    
train=train_original.copy(deep=True)

total_train=train.isnull().sum().reset_index(drop=True)
null_percentage=(train.isnull().sum()/train.isnull().count()*100).reset_index(drop=True)
columns=pd.Series(train.columns).reset_index(drop=True)
percentage_df=pd.DataFrame([columns,total_train,null_percentage]).T
percentage_df.columns=["name","total_train","null_percentage"]
percentage_df=percentage_df.sort_values(by="null_percentage", ascending=False)





####################### correlation ########################################
train_corr=train.corr()
f, ax=plt.subplots(figsize=(25,25))
mask = np.triu(np.ones_like(train_corr, dtype=bool))
sns.set(font_scale=0.75)
ax = sns.heatmap(train_corr, vmin=-1, vmax=1, mask=mask, cmap='RdBu', center=0, annot = True, square=True, linewidths=.5, cbar_kws= {"shrink": .5, 'orientation': 'vertical'})



#saleprice correlation matrix
#number of variables for heatmap
k = 15
cols = train_corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
plt.figure(figsize=(10, 16))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},cbar_kws= {"shrink": .5, 'orientation': 'vertical'}, yticklabels=cols.values,linewidths=.5, xticklabels=cols.values)
plt.show()

train=train[cols]

sns.scatterplot(train["GarageArea"],train["SalePrice"])



garageArea_outliers=train["GarageArea"].sort_values(ascending=False)[:4]

train = train.drop(train[train['GarageArea'] >1248].index)

import time

for col in cols:
    print(col)
    if col!="SalePrice":
        sns.scatterplot(train[col],train["SalePrice"])
        time.sleep(5)
    


sns.scatterplot(x=df_train.MasVnrArea,y=df_train.SalePrice)