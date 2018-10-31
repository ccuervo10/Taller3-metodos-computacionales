
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
fig.savefig(filename='ApellidoNombre_signal.pdf', type='pdf', dpi=100)

