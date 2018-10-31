
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft.fftpack import ifft
from scipy.interpolate import interp1d



D1 = np.genfromtxt('signal.dat', delimiter=',')
Dincompletos = np.genfromtxt('incompletos.dat', delimiter=',')

# Interpolaciones
nInterpolar = 512 # nuermo de puntos
# datos donde interpolar
xInterpolar = np.linspace(min(Dincompletos[:,0]), max(Dincompletos[:,0]), nInterpolar)
# Arreglos para guardar las interpolaciones
D2 = np.zeros((512,2))
D3 = np.zeros((512,2))

# Guardar los x
D2[:,0] = xInterpolar
D3[:,0] = xInterpolar

# Guardar los y
D2[:,1] = interp1d(Dincompletos[:,0], Dincompletos[:,1],'quadratic')(xInterpolar)
D3[:,1] = interp1d(Dincompletos[:,0], Dincompletos[:,1],'cubic')(xInterpolar)

# crear arreglos XY para guardar las 3 senales
X = np.zeros((nInterpolar, 3))
Y = np.zeros((nInterpolar, 3))

# Guardar todos los x en la misma matriz
X[:,0] = D1[:,0]
X[:,1] = D2[:,0]
X[:,2] = D3[:,0]


# Guardar todos los y en la misma matriz
Y[:,0] = D1[:,1]
Y[:,1] = D2[:,1]
Y[:,2] = D3[:,1]


# Grafica basica de la senal
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(X[:,0],Y[:,0])
ax.set_xlabel('x')
ax.set_xlabel('y')
fig.savefig(filename='CuervoCamilo_signal.pdf', type='pdf', dpi=100)

def fourier(Y, ncolumnas):
    # Hacer transformada de cada columna en Y
    
    Transformadas = np.zeros(np.shape(Y), dtype=complex)
    
    nfilas = np.size(Transformadas,0) # numero filas
    
    for k in range(nfilas):
        for n in range(nfilas):
            for i in range(ncolumnas):
                # Hacer la trasnformada de todas las columnas al tiempo
                Transformadas[k,i] = Transformadas[k,i] + Y[n,i]*np.exp(-1j*2*np.pi*n*k/nfilas)

    return Transformadas



def frecuencias(X, ncolumnas):
    # Encontrar vector de frecuencias para cada columna
    dx = X[1,:]-X[0,:] # delta de x
    fs = 1/dx # frecuencias de muestreo
    
    Frecuencias = np.zeros(np.shape(X), dtype=float)
    nfilas = np.size(Frecuencias, 0) # numero de filas
    
    for i in range(ncolumnas):
        # crear vector de frecuencias para cada columna
        Frecuencias[:,i] = np.linspace(-fs[i]/2, fs[i]/2, nfilas)
    
    # Rodar el arreglo de frecuencias para que coincidan con la transformada
    Frecuencias = np.roll(Frecuencias, int(nfilas/2), axis=0)
    
    return Frecuencias



def filtrar(X, Y, Xcorte, ncolumnas):
    # Filtrar ncolumnas de Y acorde a valores de X
    
    # Iniciar filtro en ceros
    Filtro = np.zeros(np.shape(Y), dtype=complex)
    nfilas = np.size(Filtro, 0) # numero de filas
    
    # Encontrar los valores que se permiten pasar y replicarlos en Filtro
    for k in range(nfilas):
        elegir_frecuencias = np.abs(X[k,:]) < np.abs(Xcorte)
        Filtro[k,elegir_frecuencias] = Y[k,elegir_frecuencias]
    
    return Filtro

def fourier(Y, ncolumnas):
    # Hacer transformada de cada columna en Y
    
    Transformadas = np.zeros(np.shape(Y), dtype=complex)
    
    nfilas = np.size(Transformadas,0) # numero filas
    
    for k in range(nfilas):
        for n in range(nfilas):
            for i in range(ncolumnas):
                # Hacer la trasnformada de todas las columnas al tiempo
                Transformadas[k,i] = Transformadas[k,i] + Y[n,i]*np.exp(-1j*2*np.pi*n*k/nfilas)

    return Transformadas



def frecuencias(X, ncolumnas):
    # Encontrar vector de frecuencias para cada columna
    dx = X[1,:]-X[0,:] # delta de x
    fs = 1/dx # frecuencias de muestreo
    
    Frecuencias = np.zeros(np.shape(X), dtype=float)
    nfilas = np.size(Frecuencias, 0) # numero de filas
    
    for i in range(ncolumnas):
        # crear vector de frecuencias para cada columna
        Frecuencias[:,i] = np.linspace(-fs[i]/2, fs[i]/2, nfilas)
    
    # Rodar el arreglo de frecuencias para que coincidan con la transformada
    Frecuencias = np.roll(Frecuencias, int(nfilas/2), axis=0)
    
    return Frecuencias



