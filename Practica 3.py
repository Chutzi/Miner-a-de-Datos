# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 21:04:06 2022

@author: chuch
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

#Practica 3 - Data Analysis

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

#Aplicando funciones de agregado
def analysis_price(df: pd.DataFrame)->pd.DataFrame:
    df_by_p = df.groupby(["MetropolitanRegion"]).agg({'Price': ['sum', 'count']})
    print_tabulate(df_by_p.head())
    df_by_p = df.groupby(["MetropolitanRegion"]).agg({'Price': ['sum', 'count', 'mean', 'min', 'max']})
    df_by_p = df_by_p.reset_index()
    print_tabulate(df_by_p.head())
    return df_by_p
    
#DataFrame normalizado
dfNorm = normalize_data(dftest)

#Análisis del precio con el DF normalizado
dfa = analysis_price(dfNorm)



