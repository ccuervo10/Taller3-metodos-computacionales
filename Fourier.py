
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


# Hacer la transformada de todas las senales
T = fourier(Y, 3)

# Encontrar las frecuencias de todas las senales
F = frecuencias(X,3)

# Filtrar con 1000 y con 500 como limite
T1000 = filtrar(F, T, 1000, 3)
T500 = filtrar(F, T, 500, 3)

# Hacer transformadas inversas
Y1000 = ifft(T1000, axis=0)
Y500 = ifft(T500, axis=0)



# Grafica de transformada de la senal
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(F[:,0],np.abs(T[:,0]))
ax.set_xlabel('frecuencias')
ax.set_xlim([-500,500])
ax.grid()
fig.savefig(filename='CuervoCamilo_TF.pdf', type='pdf', dpi=100)

# Grafica de transformada inversa de la senal despues de filtrar a 1000Hz
fig.clear()
ax = fig.gca()
ax.plot(X[:,0],np.real(Y1000[:,0]))
ax.set_xlabel('x')
fig.savefig(filename='CuervoCamilo_filtrada.pdf', type='pdf', dpi=100)

# Grafica de espectros de frecuencia de las 3 senales
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6,8))
ax[0].plot(F[:,0],np.abs(T[:,0]))
ax[0].set_title('original')
ax[1].plot(F[:,1],np.abs(T[:,1]))
ax[1].set_title('Interpolacion cuadratica')
ax[2].plot(F[:,2],np.abs(T[:,2]))
ax[2].set_title('Interpolacion cubica')
ax[2].set_xlabel('frecuencias')
fig.savefig(filename='CuervoCamilo_filtrada.pdf', type='pdf', dpi=100)

# Grafica de transformada despues de filtrar a 1000Hz y a 500Hz
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,8))
ax[0].plot(X,np.real(Y500))
ax[0].set_title('Filtro 500Hz')
ax[0].legend('123')
ax[1].plot(X,np.real(Y1000))
ax[1].set_title('Filtro 1000Hz')
ax[1].set_xlabel('frecuencias')
ax[1].legend('123')
fig.savefig(filename='CuervoCamilo_2Filtros.pdf', type='pdf', dpi=100)

print("Las frecuencias las calcule con mi propio codigo", "\n"*2)
print("Los picos de frecuencia mas importantes estan en -170 y 170 Hz, segudidos por picos en -200 y 200Hz y -400 y 400Hz", "\n"*2)
print("Los datos incompletos no tienen nigun patron en el periodo de muestreo, en general todos son diferentes. Por esa razon no tiene sentido calcular los multiplos de frecuencia 2pi*n*k de la formula de la transformada discreta", "\n"*2)
print("A medida que aumenta el orden de la interpolacion los picos de 400Hz presentan magnitudes menores en el espectro de frecuencias. En la interpolacion cubica se activa ruido entre 500Hz y 2000 Hz. En la interpolacion cuadrada se activa ruido entre 500Hz y 5000Hz", "\n"*2)



