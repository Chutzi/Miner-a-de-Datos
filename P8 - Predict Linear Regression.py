# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:32:43 2022

@author: Jesús Alexandro Hernández Rivera
"""

import requests
import io
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numbers
import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split

#Data Importing

def get_csv_from_url(url:str) -> pd.DataFrame:
    try:
        s = requests.get(url).content
        print("Extracción exitosa")
        return pd.read_csv(io.StringIO(s.decode('utf-8')))
    except:
        print("Sin conexión a internet\nSe leyó de un archivo local")
        return pd.read_csv("typed_melb_clean_data.csv")

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

def transform_variable(df: pd.DataFrame, x:str)->pd.Series:
    if isinstance(df[x][df.index[0]], numbers.Number):
        return df[x] # type: pd.Series
    else:
        return pd.Series([i for i in range(0, len(df[x]))])
    
def transform_into_typed_df(raw_df: pd.DataFrame)->pd.DataFrame:
    raw_df["Date"] = pd.to_datetime(raw_df['Date'], format="%Y/%m/%d")
    return raw_df
    
#Datos normalizados y limpios
url = "https://raw.githubusercontent.com/Chutzi/Mineria-de-Datos/master/typed_melb_clean_data.csv"
dfn = get_csv_from_url(url)
#print_tabulate(dfn.head())
dfn = transform_into_typed_df(dfn)

#Train-Test split
label = dfn.pop('Price')
label=np.log(label)
data_train, data_test, label_train, label_test = train_test_split(dfn, label, test_size = 0.2, random_state = 500)

def label_transform(df: pd.DataFrame, columns: list):
    for c in columns:
        lbl = preprocessing.LabelEncoder()
        df[c] = lbl.fit_transform(df[c].astype(str))

columns=['Suburb', 'Address', 'Type', 'Method', 'SellerG','Date',  'CouncilArea', 'MetropolitanRegion']

label_transform(data_train, columns)
label_transform(data_test, columns)

#print(data_train.dtypes)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(data_train, label_train)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

dtest=xgb.DMatrix(data_test)
y_predict = np.exp2(model.predict(dtest))
Actual_Price=np.exp2(label_test)
out = pd.DataFrame({'Actual_Price': Actual_Price, 'predict_Price': y_predict,'Diff' :(Actual_Price-y_predict)})

print(out[['Actual_Price','predict_Price','Diff']].head(10))

sns.set(color_codes=True)
#sns.regplot(out['predict_Price'],out['Diff'], line_kws={"color":"red","alpha":0.5,"lw":4}, marker="x")
sns.regplot(out['Actual_Price'],out['Diff'], line_kws={"color":"red","alpha":0.5,"lw":4}, marker="x")
plt.savefig("Regresion lineal/actual_price.png")
