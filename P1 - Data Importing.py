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

#Practica 1 - Data Importing

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
print("\nDATAFRAME:\n",df)