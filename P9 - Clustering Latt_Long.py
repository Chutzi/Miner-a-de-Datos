# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 13:18:08 2022

@author: chuch
"""

# agglomerative clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot
import requests
import io
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
from typing import Tuple, List
import re
from datetime import datetime
import numpy as np
import numbers
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.svm import SVC
from copy import deepcopy
from sklearn.cluster import KMeans

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

#Datos normalizados y limpios
url = "https://raw.githubusercontent.com/Chutzi/Mineria-de-Datos/master/typed_melb_clean_data.csv"
dfNorma = get_csv_from_url(url)
print_tabulate(dfNorma.head())

df_c=dfNorma
df_c.drop(['Price'], axis=1,inplace=True)
df_c.head()

df_imp=df_c[['Lattitude','Longtitude']]
from sklearn import preprocessing

x = df_imp #regresa un numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.head()

def Kmeans(df: pd.DataFrame(), n: int):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(df)
    labels = kmeans.labels_
    #Pegar de nuevo a los datos originales
    df['clusters'] = labels
    df2 = df.rename(columns = {0 : 'Lattitude', 1: 'Longtitude'})
    #Agregar la columna a nuestra lista
    return df2

import seaborn as sns
k=3;
sns.lmplot('Lattitude', 'Longtitude', data = Kmeans(df, k), fit_reg=False,hue="clusters",  scatter_kws={"marker": "D", "s": 100})
plt.title('Latitude v/s Longitude')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.savefig(f'Clustering/cluster_{k}_Latitude_Longitude.png')
plt.show()