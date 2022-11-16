# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 08:43:31 2022

@author: Jesús Alexandro Hernández Rivera
"""

import requests
import io
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
from typing import Tuple, List
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
        return pd.read_csv("typed_melb_clean_data.csv")

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

#Datos normalizados y limpios
url = "https://raw.githubusercontent.com/Chutzi/Mineria-de-Datos/master/typed_melb_clean_data.csv"
dfNorma =get_csv_from_url(url)
print_tabulate(dfNorma.head())

def generate_df(means: List[Tuple[float, float, str]], n: int) -> pd.DataFrame:
    lists = [
        (dfNorma["Lattitude"], dfNorma["Longtitude"], dfNorma["Type"])
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

def scatter_group_by(
    file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str
):
    fig, ax = plt.subplots(figsize=(10,6))
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f"{label_column} == '{label}'")
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i))
    ax.legend()
    #plt.plot(figsize=(4,4))
    plt.title('Latitude v/s Longitude')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.savefig(file_path)
    plt.close()

#KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(df["x"], df["y"])

groups = [(1825, 10000, "grupo1"), (1900, 31000, "grupo2"), (1975, 3.3e+06, "grupo3")]
df = generate_df(groups, 100)
filtro = df['x'] < 400000
df = df[filtro]
scatter_group_by("K Nearest Neighbors/groups_by_room_type.png", df, "x", "y", "label")

