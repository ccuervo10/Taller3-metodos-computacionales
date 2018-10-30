import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

from scipy.fftpack import *
from PIL import Image

# Cargar imagen con PIL
Imagen = Image.open('arbol.PNG')
Imagen = Imagen.convert("L")
arreglo = np.asarray(Imagen)

Fourier = fft2(arreglo) # transformada 2d

# Grafica de trasnformada de Fourier
Fourier_imagen = np.log(np.abs(Fourier))
Fourier_imagen = Fourier_imagen/np.max(Fourier_imagen)*255
Fourier_imagen = Image.fromarray(np.uint8(Fourier_imagen))
Fourier_imagen.save("CuervoCamilo_FT2D.pdf", 'pdf')

