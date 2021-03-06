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

for i in range(variables):
    for j in range(i, variables): # va desde i porque la matriz es simetrica y 
        # no vale la pena repetir los mismos calculos
        
        # Encontrar promedios de variables i, j
        promedio_i = np.mean(Cancer[:,i])
        promedio_j = np.mean(Cancer[:,j])
        covij = 0
        for k in range(observaciones):
            covij = covij  + (Cancer[k,i]-promedio_i)*(Cancer[k,i]-promedio_j)/observaciones
        matriz[i,j] = covij
        
for i in range(variables):
    for j in range(i):
        matriz[i,j] = matriz[j,i]

print("\n"*3)
print("Matriz de covarianza")
print("\n"*3)
print(matriz)


Auto = np.linalg.eig(matriz) # autovalores y autovectores con linalg

print("\n"*3)
print("Autovalor")
print("\tAutovector")
print("\t\t--- Variable mas importante")
for i in range(variables):
    print(Auto[0][i])
    for j in range(variables):
        print("\t ", Auto[1][j,i])
    print("\t\t--- ", np.argmax(np.abs(Auto[1][:,i])))



# indices para ordenar los autovalores
indices = np.argsort(Auto[0])

autovectores = []
for i in range(variables):
    # Guardar los autovectores en el orden establecido
    autovectores.append( Auto[1][:,indices[i]])

# Extraer los componentes principales
v1 = autovectores[-1]
v2 = autovectores[-2]

# juntar autovectores en una matriz
ComponentesPrincipales = np.matrix(np.asarray([v1,v2]).T)

# convertir los datos a una matriz
MatrizDatosB = np.matrix(Benigno.T)
MatrizDatosM = np.matrix(Maligno.T)

# hacer multiplicaciones de matrices para proyectar sobre componentes principales
# hago por aparte benignos y malignos
ProyeccionesB = ComponentesPrincipales.transpose()*MatrizDatosB
ProyeccionesM = ComponentesPrincipales.transpose()*MatrizDatosM

plt.scatter(ProyeccionesB[0,:], ProyeccionesB[1,:],alpha=0.5)
plt.scatter(ProyeccionesM[0,:], ProyeccionesM[1,:], c='black', alpha=0.5)
plt.xlabel('Componente Principal 1')
plt.xlabel('Componente Principal 2')
plt.title("Proyeccion de componentes principales. Maligno mostrado en negro y 50% de transparencia")
# zoom en la zona donde se traslapan
plt.xlim([-2000,0])
plt.ylim([-400,0])
plt.savefig('CuervoCamilo_PCA.pdf')

print("En la imagen de las proyecciones que se presenta se ha hecho un zoom en la zona de interes para demostrar que los diagnosticos benignos se aglomeran en una zona definida, aunque las fronteras de esta zona son difusas entre puntos de diagnostico benigno y maligno. En general esta grafica muestra que hay potencial para usar PCA como una herramienta para evaluar si las mediciones indican un tumor Benigno o maligno. Pero tratandose de un tema tan delicado e importante, la informacion suministrada no es suficiente para desarrollar una herramienta confiable")
