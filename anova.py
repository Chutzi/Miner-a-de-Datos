# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 21:00:34 2022

@author: chuch
"""

import requests
import io
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#Data Importing

def get_csv_from_url(url:str) -> pd.DataFrame:
    try:
        s = requests.get(url).content
        print("Extracción exitosa")
        return pd.read_csv(io.StringIO(s.decode('utf-8')))
    except:
        print("Sin conexión a internet\nSe leyó de un archivo local")
        return pd.read_csv("melb_data.csv")

url = "https://raw.githubusercontent.com/Chutzi/Mineria-de-Datos/master/typed_melb_clean_data.csv"
df = get_csv_from_url(url)

voter_frame = pd.DataFrame({"Price":df["Price"],"Region":df["Landsize"]})

model = ols('Region ~ Price',         # Model formula
            data = voter_frame).fit()
                
anova_result = sm.stats.anova_lm(model, typ=2)
if anova_result["PR(>F)"][0] < 0.005:
    print("\nANOVA: hay diferencias")
    print(anova_result, "\n")
    # Prueba tukey
    tukey = pairwise_tukeyhsd(endog=df["Price"],     # Data
                              groups=df["MetropolitanRegion"],   # Groups
                              alpha=0.05)          # Significance level

    tukey.plot_simultaneous()    # Plot group confidence intervals
    plt.vlines(x=49.57,ymin=-0.5,ymax=4.5, color="red")
    plt.show()
    print(tukey.summary())              # See test summary
    # imprimir los resultados
else:
    print("No hay diferencias")




