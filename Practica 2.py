# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:30:56 2022

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
print("\nDATAFRAME\n",df)

#Practica 2 - Data Cleaning -----------------------------------------

#Elimina los registros que contengan valores nulos (0), o columnas (1)
dfTest = df.dropna(0)
print("\nTest\n",dfTest)

#Llenar los datos nulos con la media de la tabla correspondiente
df['BuildingArea'].fillna(value=int(df['BuildingArea'].mean()), inplace=True)
df['YearBuilt'].fillna(value=int(df['YearBuilt'].mean()), inplace=True)
df['Car'].fillna(value=int(df['Car'].mean()), inplace=True)
df['CouncilArea'].fillna(value='Unknown', inplace=True)

#Ordena por la columna de forma ascendente 
df.sort_values(by="Suburb", ascending=True)

#Agrupa las suburbs con la media de cada columna del dataframe  
suburbs = df.groupby(['Suburb']).mean()
print("\nSUBURBS\n",suburbs)

#Suma los valores de la columnas e imprime el precio total 
suma = suburbs.apply(np.sum, axis=0)
print("\nPRECIO TOTAL:\n",suma[1:2])

#Convertir dataframe a un CSV limpio 
df.to_csv('melb_clean_data.csv')

#Data Statistics

print("\nMuestra de Registros\n")

#Primeros 5 registros
print(df.head())

#Ultimos 5 registros
print(df.tail())

#Información general del dataframe 
print(df.info())


#Informa datos estadísticos
print(df.describe())

#Media, mediana y correlación de la columna Price
print('Promedio general de los precios: $',df['Price'].mean())
print('Mediana General de los precios:',df['Price'].median())
print('Desviación estándar General de los precios:',df['Price'].std())

df2 = df.corr()
print("Correlacion es:\n", df2)

#Cantidad de valores en filas
print("\nCantidad de valores\n",df.count())

#Devuelve el máximo de los valores sobre el eje solicitado.
print("\nMaximo:\n",df.max())

#Devuelve el mínimo de los valores sobre el eje solicitado.
print("\nMinimo:\n",df.min())