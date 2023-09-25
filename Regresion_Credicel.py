#Regresión Simple Credicel

#Librerías
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

#Carga de archivos
cred = pd.read_csv("Credicel_Limpio3.csv")

print(cred.columns())

#División de plazos
print(cred["plazo"].value_counts())

col_cred = ["precio", "enganche", "monto_financiado", "costo_total", "plazo"]
cred1 = cred[col_cred]

semana_13 = cred1[(cred1["plazo"] == "13s")]
semana_26 = cred1[(cred1["plazo"] == "26s")]
semana_39 = cred1[(cred1["plazo"] == "39s")]
semana_52 = cred1[(cred1["plazo"] == "52s")]


