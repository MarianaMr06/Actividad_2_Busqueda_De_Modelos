# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Carga de archivos
cred = pd.read_csv("Credicel_Limpio3.csv")

col_cred = ["precio", "enganche", "monto_financiado", "costo_total", "plazo",
            "pagos_realizados",
            "semana"]
cred1 = cred[col_cred]

semana_13 = cred1[(cred1["plazo"] == "13S")]
semana_26 = cred1[(cred1["plazo"] == "26S")]
semana_39 = cred1[(cred1["plazo"] == "39S")]
semana_52 = cred1[(cred1["plazo"] == "52S")]

#Declaración de variables semana 13
x_enganche_13 = semana_13["enganche"]
x_monto_financiado_13 = semana_13["monto_financiado"]
x_costo_total_13 = semana_13["costo_total"]
x_semana_13 = semana_13["semana"]
x_pagos_realizados_13 = semana_13["pagos_realizados"]
y_precio_13 = semana_13["precio"]
y_costo_total_13 = semana_13["costo_total"]

print(x_costo_total_13)

#Declaración de variables semana 26
x_enganche_26 = semana_26["enganche"]
x_monto_financiado_26 = semana_26["monto_financiado"]
x_costo_total_26 = semana_26["costo_total"]
x_semana_26 = semana_26["semana"]
x_pagos_realizados_26 = semana_26["pagos_realizados"]
y_precio_26 = semana_26["precio"]
y_costo_total_26 = semana_26["costo_total"]

#Declaración de variables semana 39
x_enganche_39 = semana_39["enganche"]
x_monto_financiado_39 = semana_39["monto_financiado"]
x_costo_total_39 = semana_39["costo_total"]
x_semana_39 = semana_39["semana"]
x_pagos_realizados_39 = semana_39["pagos_realizados"]
y_precio_39 = semana_39["precio"]
y_costo_total_39 = semana_39["costo_total"]

#Declaración de variables semana 52
x_enganche_52 = semana_52["enganche"]
x_monto_financiado_52 = semana_52["monto_financiado"]
x_costo_total_52 = semana_52["costo_total"]
x_semana_52 = semana_52["semana"]
x_pagos_realizados_52 = semana_52["pagos_realizados"]
y_precio_52 = semana_52["precio"]
y_costo_total_52 = semana_52["costo_total"]

# Funciones
def correlacion_determinacion(variableX, variableY, df):
    model = LinearRegression()
    
    if isinstance(variableX, pd.DataFrame):
        # Si variableX es un DataFrame, conviértelo a una Serie de pandas o un arreglo de numpy unidimensional
        variableX = variableX.squeeze()
    
    # Convierte variableX a una lista
    lista_variableX = variableX.tolist()
    
    # Asegurar que variableY sea un arreglo de numpy unidimensional
    if isinstance(variableY, pd.Series):
        variableY = variableY.values
    
    # Reshape para una sola característica (lista_variableX es una lista)
    X = np.array(lista_variableX).reshape(-1, 1)
    model.fit(X, variableY)
    # Calcular coeficiente de determinación
    coef_determinacion = model.score(X, variableY)
    # Calcular coeficiente de correlación
    coef_correlacion = np.sqrt(coef_determinacion)
    # Mostrar coeficientes de determinación y correlación
    print("Coeficiente de determinación:", coef_determinacion)
    print("Coeficiente de correlación:", coef_correlacion)
    # Graficar dispersión
    sns.scatterplot(x=lista_variableX, y=variableY, color='purple')
    plt.xlabel('variableX')
    plt.ylabel('variableY')
    plt.show()

    
# Semana 13

# Precio vs Enganche
precio_enganche_13 = correlacion_determinacion(semana_13["enganche"], semana_13["precio"], cred)

# Precio vs Monto Financiado
precio_montofinanciado_13 = correlacion_determinacion(semana_13["monto_financiado"], semana_13["precio"], cred)

# Precio vs Costo Total
precio_costototal_13 = correlacion_determinacion(semana_13["costo_total"], semana_13["precio"], cred)

# Costo total vs Monto Financiado
costototal_montofinanciado_13 = correlacion_determinacion(semana_13["monto_financiado"], semana_13["costo_total"], cred)

# Precio vs semana
precio_semana_13 = correlacion_determinacion(semana_13["semana"], semana_13["precio"], cred)

# Precio vs Pagos Realizados
precio_pagosrealizados_13 = correlacion_determinacion(semana_13["pagos_realizados"], semana_13["precio"], cred)

# Semana 26

# Precio vs Enganche
precio_enganche_26 = correlacion_determinacion(semana_26["enganche"], semana_26["precio"], cred)

# Precio vs Monto Financiado
precio_montofinanciado_26 = correlacion_determinacion(semana_26["monto_financiado"], semana_26["precio"], cred)

# Precio vs Costo Total
precio_costototal_26 = correlacion_determinacion(semana_26["costo_total"], semana_26["precio"], cred)

# Costo total vs Monto Financiado
costototal_montofinanciado_26 = correlacion_determinacion(semana_26["monto_financiado"], semana_26["costo_total"], cred)

# Precio vs semana
precio_semana_26 = correlacion_determinacion(semana_26["semana"], semana_26["precio"], cred)

# Precio vs Pagos Realizados
precio_pagosrealizados_26 = correlacion_determinacion(semana_26["pagos_realizados"], semana_26["precio"], cred)

# Semana 39

# Precio vs Enganche
precio_enganche_39 = correlacion_determinacion(semana_39["enganche"], semana_39["precio"], cred)

# Precio vs Monto Financiado
precio_montofinanciado_39 = correlacion_determinacion(semana_39["monto_financiado"], semana_39["precio"], cred)

# Precio vs Costo Total
precio_costototal_39 = correlacion_determinacion(semana_39["costo_total"], semana_39["precio"], cred)

# Costo total vs Monto Financiado
costototal_montofinanciado_39 = correlacion_determinacion(semana_39["monto_financiado"], semana_39["costo_total"], cred)

# Precio vs semana
precio_semana_39 = correlacion_determinacion(semana_39["semana"], semana_39["precio"], cred)

# Precio vs Pagos Realizados
precio_pagosrealizados_39 = correlacion_determinacion(semana_39["pagos_realizados"], semana_39["precio"], cred)

# Semana 52

# Precio vs Enganche
precio_enganche_52 = correlacion_determinacion(semana_52["enganche"], semana_52["precio"], cred)

# Precio vs Monto Financiado
precio_montofinanciado_52 = correlacion_determinacion(semana_52["monto_financiado"], semana_52["precio"], cred)

# Precio vs Costo Total
precio_costototal_52 = correlacion_determinacion(semana_52["costo_total"], semana_52["precio"], cred)

# Costo total vs Monto Financiado
costototal_montofinanciado_52 = correlacion_determinacion(semana_52["monto_financiado"], semana_52["costo_total"], cred)

# Precio vs semana
precio_semana_52 = correlacion_determinacion(semana_52["semana"], semana_52["precio"], cred)

# Precio vs Pagos Realizados
precio_pagosrealizados_52 = correlacion_determinacion(semana_52["pagos_realizados"], semana_52["precio"], cred)
