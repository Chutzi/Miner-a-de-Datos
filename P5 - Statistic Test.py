# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:56:08 2022

@author: Jesus Alexandro Hernández Rivera 
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
import statsmodels.api as sm
from statsmodels.formula.api import ols

#Data Importing

def get_csv_from_url(url:str) -> pd.DataFrame:
    try:
        s = requests.get(url).content
        print("Extracción exitosa")
        return pd.read_csv(io.StringIO(s.decode('utf-8')))
    except:
        print("Sin conexión a internet\nSe leyó de un archivo local")
        return pd.read_csv("melb_data.csv")

url = "https://raw.githubusercontent.com/Chutzi/Datasets/main/melb_data.csv"
df = get_csv_from_url(url)
#print("\nDATAFRAME\n",df)

#Data Cleaning with Data Statistics

#Llenar los datos nulos con la media de la tabla correspondiente
df['BuildingArea'].fillna(value=int(df['BuildingArea'].mean()), inplace=True)
df['YearBuilt'].fillna(value=int(df['YearBuilt'].mean()), inplace=True)
df['Car'].fillna(value=int(df['Car'].mean()), inplace=True)
df['CouncilArea'].fillna(value='Unknown', inplace=True)

#Ordena por la columna de forma ascendente 
df.sort_values(by="Price", ascending=True)

#Convertir dataframe a un CSV limpio 
df.to_csv('melb_clean_data.csv')

#Data Analysis

#Importar en consola
def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

#Cambiar el nombre de la columna que usaremos
dftest = df.rename(columns={'Regionname': 'MetropolitanRegion'})

#Categorización de la región metropolitana a la que pertenece
def categorize(name:str)->str:
    if 'South-Eastern Metropolitan' in name:
        return 'South-Eastern'
    if 'Northern Metropolitan' in name:
        return 'Northern'
    if 'Western Metropolitan' in name:
        return 'Western'
    if 'Southern Metropolitan' in name:
        return 'Southern'
    if 'Eastern Metropolitan' in name:
        return 'Eastern'
    return 'Other'

def transform_into_typed_df(raw_df: pd.DataFrame)->pd.DataFrame:
    raw_df["Date"] = pd.to_datetime(raw_df['Date'], format="%d/%m/%Y")
    raw_df["MetropolitanRegion"] = raw_df["MetropolitanRegion"].map(categorize)
    return raw_df

#Normalización de los datos
def normalize_data(df: pd.DataFrame)->pd.DataFrame:
    df_complete = transform_into_typed_df(df)
    df_complete.to_csv("typed_melb_clean_data.csv", index=False)
    return df_complete
    
def plot_by_average_price_per_region(df: pd.DataFrame)->None:
    plt.title("Average price of each Metropolitan Region")
    plt.xlabel("Metropolitan Region")
    plt.ylabel("Price")
    plt.plot(df["MetropolitanRegion"], df["Price","mean"].sort_values())
    plt.savefig(f"img/average_price_per_region.png")
    plt.show()
  
def create_plot(df: pd.DataFrame):
   df.reset_index(inplace=True)
   df.set_index("Date", inplace=True)
   print_tabulate(df.head(5))

   for dep in set(df["MetropolitanRegion"]):
     plot_by_dep(df, dep)
   df_aux = df.groupby(["Date","MetropolitanRegion"])[['Price']].mean().unstack()
   df_aux.plot(y = 'Price', legend=False, figsize=(32,18))
   plt.xticks(rotation=90)
   plt.savefig("img/foo.png")
   plt.close()

def plot_by_dep(df: pd.DataFrame, dep:str)->None:
    df[df["MetropolitanRegion"] == dep].plot(y =["Price"], figsize=(32,18))
    plt.title(dep)
    plt.savefig(f"img/lt_{dep}.png")
    df[df["MetropolitanRegion"] == dep].boxplot(by ='Price', figsize=(32,18))
    plt.savefig(f"img/bplt_{dep}.png")
    
def create_boxplot_by_type(df: pd.DataFrame, column: str, agg_fn=pd.DataFrame.sum):
    df_by_type = df.groupby([column,"Date"])[["Price"]].aggregate(agg_fn)#.count()
    df_by_type.boxplot(by = column, figsize=(54,36))
    plt.xticks(rotation=90)
    plt.savefig(f"img/boxplot_{column}.png")
    plt.close()
    
def anova(df_aux: pd.DataFrame, str_ols: str):
    # shaphiro-wills
    # Levenes or barletts
    modl = ols(str_ols, data=df_aux).fit()
    anova_df = sm.stats.anova_lm(modl, typ=2)
    if anova_df["PR(>F)"][0] < 0.005:
        print("hay diferencias")
        print(anova_df)
        # Prueba tukey
        # imprimir los resultados
    else:
        print("No hay diferencias")

def anova_1(df_complete: pd.DataFrame):
    df_by_type = df_complete.groupby(["Suburb", "Date"])[["Price"]].aggregate(pd.DataFrame.sum)
    df_by_type.reset_index(inplace=True)
    df_by_type.set_index("Date", inplace=True)
    df_by_type.reset_index(inplace=True)
    df_aux = df_by_type.rename(columns={"Price": "Gasto"}).drop(['Date'], axis=1)
    print(df_aux.head())
    anova(df_aux, "Gasto ~ Suburb")
    

#DataFrame normalizado
dfNorm = normalize_data(dftest)

#Análisis del precio con el DF normalizado
#dfa = analysis_price(dfNorm)

#create_plot(dfNorm)

#Grafica BoxPlot del precio según cada suburb
#create_boxplot_by_type(dfNorm, "Suburb", pd.DataFrame.mean)

#ANOVA
anova_1(dfNorm)

