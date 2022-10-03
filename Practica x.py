# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 11:34:44 2022

@author: Jesús Alexandro Hernández Rivera
"""

import requests
import io
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
from typing import Tuple, List
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import numbers
import statsmodels.api as sm

#Data Importing

def get_csv_from_url(url:str) -> pd.DataFrame:
    try:
        s = requests.get(url).content
        print("Extracción exitosa")
        return pd.read_csv(io.StringIO(s.decode('utf-8')))
    except:
        print("Sin conexión a internet\nSe leyó de un archivo local")
        return pd.read_csv("melb_data.csv")

url = "https://raw.githubusercontent.com/Chutzi/Datasets/main/typed_melb_clean_data.csv"
df = get_csv_from_url(url)
#print("\nDATAFRAME\n",df)

#Importar en consola
def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

def transform_into_typed_df(raw_df: pd.DataFrame)->pd.DataFrame:
    raw_df["Date"] = pd.to_datetime(raw_df['Date'], format="%d/%m/%Y")
    return raw_df

#Practica 6 - Linear Regression

def transform_variable(df: pd.DataFrame, x:str)->pd.Series:
    if isinstance(df[x][0], numbers.Number):
        return df[x] # type: pd.Series
    else:
        return pd.Series([i for i in range(0, len(df[x]))])

def linear_regression(df: pd.DataFrame, x:str, y: str)->None:
    fixed_x = transform_variable(df, x)
    model = sm.OLS(df[y],sm.add_constant(fixed_x)).fit()
    print(model.summary())

    coef = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']
    df.plot(x=x,y=y, kind='scatter', figsize=(22,14))
    plt.plot(df[x],[pd.DataFrame.mean(df[y]) for _ in fixed_x.items()], color='green')
    plt.plot(df_by_price[x],[ coef.values[1] * x + coef.values[0] for _, x in fixed_x.items()], color='red')
    plt.xticks(rotation=90)
    plt.savefig(f'img/lr_{y}_{x}.png')
    plt.close()

#DataFrame normalizado
dfNorm = get_csv_from_url(df)

#print_tabulate(dfNorm.head(50))
dfNorm = transform_into_typed_df(dfNorm)
df_by_price = dfNorm.groupby("Date")\
              .aggregate(Price=pd.NamedAgg(column="Price", aggfunc=pd.DataFrame.mean))
df_by_price["Price"] = df_by_price["Price"]**10

df_by_price.reset_index(inplace=True)
print_tabulate(df_by_price.head())
linear_regression(df_by_price, "Date", "Price")