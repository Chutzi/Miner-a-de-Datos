# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:02:26 2022

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

def generate_df(means: List[Tuple[float, float, str]], n: int) -> pd.DataFrame:
    lists = [
        (dfNorma["Landsize"], dfNorma["Price"], dfNorma["Type"])
        for _x, _y, _l in means
    ]
    x = np.array([])
    y = np.array([])
    labels = np.array([])
    for _x, _y, _l in lists:
        x = np.concatenate((x, _x), axis=None)
        y = np.concatenate((y, _y))
        labels = np.concatenate((labels, _l))
    return pd.DataFrame({"x": x, "y": y, "label": labels})

def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)



groups = [(1825, 10000, "grupo1"), (1900, 31000, "grupo2"), (1975, 3.3e+06, "grupo3")]
df = generate_df(groups, 100)

dfC = pd.DataFrame(df, columns=['x', 'y'])

kmeans = KMeans(n_clusters=3).fit(dfC)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()
plt.savefig("img/Clustering_groups_by_room_type.png")
plt.close()
