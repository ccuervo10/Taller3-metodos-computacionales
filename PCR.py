from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt



cancer = read_csv("WDBC.dat", header=None)

# lista de variables a extraer
# la variable 0 es una identificacion del paciente entonces no es relevante
# La variable 1 es el diagnostico, Benigno-Maligno
# por eso tomo de la 2 hasta la ultima -> en total son 30 variables
v = []
for i in range(2,32):
    v.append(i)

# Indices 
Benigno = cancer[1]=='B'
Maligno = cancer[1]=='M'

# Extraer datos
Benigno = np.asarray(cancer[Benigno][v])
Maligno = np.asarray(cancer[Maligno][v])
Cancer = np.asarray(cancer[v])

observaciones = np.size(Cancer,0)
variables = np.size(Cancer,1)

matriz = np.zeros((variables,variables))
