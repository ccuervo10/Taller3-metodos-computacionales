import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

from scipy.fftpack import *
from PIL import Image

# Cargar imagen con PIL
Imagen = Image.open('Arboles.png')
Imagen = Imagen.convert("L")
arreglo = np.asarray(Imagen)

Fourier = fft2(arreglo) # transformada 2d

# Grafica de trasnformada de Fourier
Fourier_imagen = np.log(np.abs(Fourier))
Fourier_imagen = Fourier_imagen/np.max(Fourier_imagen)*255
Fourier_imagen = Image.fromarray(np.uint8(Fourier_imagen))
Fourier_imagen.save("CuervoCamilo_FT2D.pdf", 'pdf')

alpha = np.pi/4 - np.pi/64 # Angulo de elipse que voy a ubicar en el filtro

for i in range(np.size(Fourier,0)):
    for j in range(np.size(Fourier,1)):
        if (((i-128)*np.cos(alpha)+(j-128)*np.sin(alpha))/3.2)**2 + (((i-128)*np.sin(alpha)-(j-128)*np.cos(alpha))/0.3)**2 < 50**2:
            # Ecuacion de una elipse rotada y desplazada que cubre la zona de interes
            Fourier[i,j] = Fourier[i,j]/1000
