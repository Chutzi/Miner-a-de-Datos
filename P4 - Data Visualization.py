# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 21:42:43 2022

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
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as io
io.renderers.default='browser'

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

#Aplicando funciones de agregado
def analysis_price(df: pd.DataFrame)->pd.DataFrame:
    df_by_p = df.groupby(["MetropolitanRegion"]).agg({'Price': ['sum', 'count']})
    print_tabulate(df_by_p.head())
    df_by_p = df.groupby(["MetropolitanRegion"]).agg({'Price': ['sum', 'count', 'mean', 'min', 'max']})
    df_by_p = df_by_p.reset_index()
    print_tabulate(df_by_p.head())
    return df_by_p

#Visualización de los datos 

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

   for reg in set(df["MetropolitanRegion"]):
     plot_by_dep(df, reg)
   df_aux = df.groupby(["Date","MetropolitanRegion"])[['Price']].mean().unstack()
   df_aux.plot(y = 'Price', legend=False, figsize=(32,18))
   plt.xticks(rotation=90)
   plt.savefig("img/foo.png")
   plt.close()

def plot_by_dep(df: pd.DataFrame, reg:str)->None:
    df[df["MetropolitanRegion"] == reg].plot(y =["Price"], figsize=(32,18))
    plt.title(reg)
    plt.savefig(f"img/lt_{reg}.png")
    df[df["MetropolitanRegion"] == reg].boxplot(by ='Price', figsize=(32,18))
    plt.savefig(f"img/bplt_{reg}.png")
    
def create_boxplot_by_type(df: pd.DataFrame, column: str, agg_fn=pd.DataFrame.sum):
    df_by_type = df.groupby([column,"Date"])[["Price"]].aggregate(agg_fn)#.count()
    df_by_type.boxplot(by = column, figsize=(54,36))
    plt.xticks(rotation=90)
    plt.savefig(f"img/boxplot_{column}.png")
    plt.close()
    
def boxplot_by(df, column: str, agg_fn = pd.DataFrame.mean):
    df_review = df.groupby(column).aggregate(agg_fn)["Propertycount"]
    plt.boxplot(df_review)

def boxplot_por_tipo(dataframe,column: str, agg_fn = pd.DataFrame.sum):
    df = dataframe
    df_by_type = df.groupby([column,"Date"]).aggregate(agg_fn)
    print(df_by_type)#eliminar columnas que no quiero xd
    df_by_type.boxplot(by = column, figsize=(30,19))
    plt.savefig(f"img/boxplots_{column}.png")
    plt.close()

def distribution(df: pd.DataFrame, var: str):
    fig = px.histogram(df, x=var)
    fig.show()
    fig.write_image("Visualizar_Info/Distribution by suburb.png")

def distribution_regioname(df: pd.DataFrame):
    fig = px.histogram(df, color="Type", x='MetropolitanRegion')
    fig.show()
    fig.write_image("Visualizar_Info/Distribution by Regioname.png")

def analysis_boxplot(df: pd.DataFrame):
    fig = px.box(df, y="Price",x='Type')
    fig.show()
    fig.write_image("Visualizar_Info/Boxplot House type and price.png")

def analysis_boxplot_by_year(df: pd.DataFrame):
    fig = px.box(df, y="Price",x='Type',color='Year')
    fig.show()
    fig.write_image("Visualizar_Info/Boxplot years.png")

def salesPrice_suburb(df: pd.DataFrame):
    a=df.groupby(['Suburb','Lattitude','Longtitude'],as_index=False)['Price'].median()
    fig = px.scatter_mapbox(a,
                            lat="Lattitude",
                            lon="Longtitude",
                            color='Price',
                            mapbox_style='open-street-map',
                            hover_name='Suburb',
                            size='Price',
                            center={'lat': -37.8136, 'lon': 144.9631},
                            zoom=13,
                            hover_data=['Suburb','Price'],
                            title= 'SalesPrice In Each Suburb')
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_geos(projection_type="orthographic")
    fig.update_layout(template='plotly_dark',margin=dict(l=20,r=20,t=40,b=20))
    fig.show()
    fig.write_image("Visualizar_Info/Sales price in each suburb.png")

def analysis_boxplot_rooms_price(df: pd.DataFrame):
    sns.boxplot(x='Rooms',y='Price',data=df)
    plt.savefig("Visualizar_Info/Boxplot rooms-price.png")
    
def analysis_boxplot_method_price(df: pd.DataFrame):
    sns.boxplot(x='Method',y='Price',data=df)
    plt.savefig("Visualizar_Info/Boxplot Method-price.png")
    
def analysis_boxplot_region_price(df: pd.DataFrame):
    sns.boxplot(x='MetropolitanRegion',y='Price',data=df)
    plt.savefig("Visualizar_Info/Boxplot region-price.png")
    
#DataFrame normalizado
dfNorm = normalize_data(dftest)

#Análisis del precio con el DF normalizado
dfa = analysis_price(dfNorm)

#create_plot(dfNorm)

#Grafica BoxPlot del precio según cada suburb
#create_boxplot_by_type(dfNorm, "Suburb", pd.DataFrame.mean)

#Boxplot por columnas
boxplot_by(dfNorm, "MetropolitanRegion")
boxplot_by(dfNorm, "Suburb")

df_by_type = dfNorm.groupby(["MetropolitanRegion","Date"]).sum()
boxplot_por_tipo(dfNorm,"MetropolitanRegion")

#dfNorm.groupby("MetropolitanRegion").sum().boxplot(figsize=(15,6))

#distribution(dfNorm, "Suburb")
#distribution_regioname(dfNorm)
#analysis_boxplot(dfNorm)
#analysis_boxplot_by_year(dfNorm)
#salesPrice_suburb(dfNorm)
#analysis_boxplot_rooms_price(dfNorm)
#analysis_boxplot_method_price(dfNorm)
analysis_boxplot_region_price(dfNorm)

