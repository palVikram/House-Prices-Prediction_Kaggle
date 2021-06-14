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
train_original["LotFrontage"]=train_original.groupby("Neighborhood").transform(lambda x:x.fillna(x.median()))

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


### we have selected k number of variables through highest correlation (refer line 115)
train=train[cols]

train.columns


##### Removing outliers
sns.scatterplot(train["GarageArea"],train["SalePrice"])
garageArea_outliers=train["GarageArea"].sort_values(ascending=False)[:4]
train = train.drop(train[train['GarageArea'] >1200].index)
### after removing outliers 
sns.scatterplot(train["GarageArea"],train["SalePrice"])


### before removing outliers 
sns.scatterplot(x=train.MasVnrArea,y=train.SalePrice)
train = train.drop(train[train['MasVnrArea'] >1250].index)
### after removing outliers 
sns.scatterplot(x=train.MasVnrArea,y=train.SalePrice)


### before removing outliers 
sns.scatterplot(x=train.TotRmsAbvGrd,y=train.SalePrice)
train = train.drop(train[train['TotRmsAbvGrd'] >13].index)
### after removing outliers 
sns.scatterplot(x=train.TotRmsAbvGrd,y=train.SalePrice)



### before removing outliers 
sns.scatterplot(x=train.GrLivArea,y=train.SalePrice)
train = train.drop(train[train['GrLivArea'] >4000].index)
### after removing outliers 

sns.scatterplot(x=train.GrLivArea,y=train.SalePrice)



### before removing outliers 
sns.scatterplot(x=train.GarageCars,y=train.SalePrice)
train = train.drop(train[train['GarageCars'] >3].index)
### after removing outliers 

sns.scatterplot(x=train.GarageCars,y=train.SalePrice)


### before removing outliers 
sns.scatterplot(x=train.TotalBsmtSF,y=train.SalePrice)
train = train.drop(train[train['TotalBsmtSF'] >3000].index)
### after removing outliers 

sns.scatterplot(x=train.TotalBsmtSF,y=train.SalePrice)


### Normalize 

sns.displot(train["GrLivArea"], stat="density")


train["GrLivArea"]=np.log(train["GrLivArea"])


sns.displot(train["GrLivArea"], stat="density")


train.columns


train['TotalSF'] = train['TotalBsmtSF']+train['1stFlrSF']


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn import metrics


y_train=train["SalePrice"]
X_train=train.drop(['SalePrice'], axis=1, inplace=True)
X_train, Model_X_test, y_train, Model_y_test = train(X_train, y_train, test_size=0.2, random_state=42)



LinearReg = LinearRegression()
LinearReg.fit(X_train, y_train)
y_predict_train_linear=LinearReg.predict(X_train)
y_predict_test_linear= LinearReg.predict(Model_X_test)

print("Accuracy on Traing set   : ",LinearReg.score(X_train,y_train))
print("Accuracy on Testing set  : ",LinearReg.score(Model_X_test,Model_y_test))
print("__________________________________________")
print("\t\tError Table")
print('Mean Absolute Error      : ', metrics.mean_absolute_error(Model_y_test, y_predict_test_linear))
print('Mean Squared Error       : ', metrics.mean_squared_error(Model_y_test, y_predict_test_linear))
print('Root Mean Squared Error  : ', np.sqrt(metrics.mean_squared_error(Model_y_test, y_predict_test_linear)))
print('R Squared Error          : ', metrics.r2_score(Model_y_test, y_predict_test_linear))


plt.scatter(y_train, y_predict_train_linear,alpha=0.3)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('Train Real vs Train Predicted')


plt.scatter(Model_y_test, y_predict_test_linear,alpha=0.3)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('Test Real vs Test Predicted')


