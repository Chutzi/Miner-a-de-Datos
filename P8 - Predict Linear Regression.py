# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 21:38:47 2022

@author: Jesús Alexandro Hernández Rivera
"""

import requests
import io
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numbers
import pandas as pd
from tabulate import tabulate
from statsmodels.stats.outliers_influence import summary_table
from typing import Tuple, Dict
import numpy as np

#Data Importing

def get_csv_from_url(url:str) -> pd.DataFrame:
    try:
        s = requests.get(url).content
        print("Extracción exitosa")
        return pd.read_csv(io.StringIO(s.decode('utf-8')))
    except:
        print("Sin conexión a internet\nSe leyó de un archivo local")
        return pd.read_csv("typed_melb_clean_data.csv")

#Datos normalizados y limpios
url = "https://raw.githubusercontent.com/Chutzi/Mineria-de-Datos/master/typed_melb_clean_data.csv"
data = get_csv_from_url(url)
data['Date'] = pd.to_datetime(data['Date'])
print(data.head())

#We are dividing the data into predictors(INDEPENDENT) and prediction(DEPENDENT) arrays
predict=['Price']
y=data[predict]
predictors=['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X=data[predictors]

#using decision tree model
from sklearn.tree import DecisionTreeRegressor as dt
model = dt()
model.fit(X, y)

model.predict(X.head())#predicting our independent variable for first 5 roes
model.predict(X.tail())
model.predict([[2,1.0,156.0,79.0,1900.0,-37.8079,144.9934]]) #prediction for custom input

#MODEL VALIDATION using mean absolute error
from sklearn.metrics import mean_absolute_error
predict=model.predict(X)
mean_absolute_error(y,predict)

#MODEL VALIDATION by dividing data into training and tesing dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as dt

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
melbourne_model =dt()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

#OPtimium fitting using max_leaf_nodes as a meetric for finding out optimum number of leaves to use
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    
#Using Random forrests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y.values.ravel())
#The function expects train_y to be a 1D array,ravel() converts the 2d array to 1d array 
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
train_X = sm.add_constant(train_X, prepend=True)
modelo = sm.OLS(endog=train_y, exog=train_X,)
modelo = modelo.fit()
print(modelo.summary())
predicciones = modelo.get_prediction(exog = train_X).summary_frame(alpha=0.05)
predicciones.head(4)

# Intervalos de confianza para los coeficientes del modelo
modelo.conf_int(alpha=0.05)

# Predicciones con intervalo de confianza del 95%
predicciones = modelo.get_prediction(exog = train_X).summary_frame(alpha=0.05)
predicciones.head(4)

# Predicciones con intervalo de confianza del 95%
predicciones = modelo.get_prediction(exog = train_X).summary_frame(alpha=0.05)
predicciones['x'] = train_X["YearBuilt"]
predicciones['y'] = train_y
predicciones = predicciones.sort_values('x')

# Gráfico del modelo
fig, ax = plt.subplots(figsize=(25, 10))

ax.scatter(predicciones['x'], predicciones['y'], marker='o', color = "gray")
ax.plot(predicciones['x'], predicciones["mean"], linestyle='-', label="OLS")
ax.plot(predicciones['x'], predicciones["mean_ci_lower"], linestyle='--', color='red', label="95% CI")
ax.plot(predicciones['x'], predicciones["mean_ci_upper"], linestyle='--', color='red')
ax.fill_between(predicciones['x'], predicciones["mean_ci_lower"], predicciones["mean_ci_upper"], alpha=0.1)
ax.legend();
fig.savefig('img/predict_lr_Price_Year.png')
