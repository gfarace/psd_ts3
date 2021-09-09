#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/8/2021

@author: Gisela Farace

Descripción: Tarea semanal 2
Desarrollar un algoritmo que calcule la transformada discreta de Fourier (DFT)
------------
"""

# Importación de módulos para Jupyter

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig
#import pdsmodulos as pds

#%% Funciones

# Senoidal

def senoidal(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs # tiempo de muestreo
    df = fs/nn # resolución espectral 
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (nn-1)*ts, nn).flatten()
    
    # grilla de sampleo frecuencial
    sen = vmax*np.sin(2*np.pi*ff*tt + ph)+dc
    
    return tt, sen

# La DFT se define como:
# X_k = sumatoria{x_n.e^(-j2.pi.k.n/N)} desde n=0 a N-1

def dft(xx):
    N = xx.size
    n = np.arange(N)    
    k = n.reshape((N, 1)) 
    XX = np.zeros(N,dtype=np.int8) # Matriz de Nx1 ceros

    exponencial = np.exp(-2j*np.pi*k*n/N)
    
    XX = np.dot(exponencial, xx) #Producto de dos arrays

    return  XX 
#%% Senoidal
# Parámetros
# vmax = 1
# dc = 0
# ff = 1
# ph = 0
# nn = 1000
# fs = 100 

# tt,xx = senoidal(vmax, dc, ff, ph, nn, fs)

# plt.figure(1)
# line_hdls = plt.plot(tt, xx)
# plt.title('Señal senoidal')
# plt.xlabel('tiempo [seg]')
# plt.ylabel('Amplitud [V]')

# axes_hdl = plt.gca()

# plt.show()

#%% Usando la DFT rápida FFT

# xx_fft = np.fft.fft(xx)

# N = xx_fft.size
# df = 1/(N*(1/fs))
# freq = np.linspace(0, (N-1), num=N)*df

# plt.figure(2)
# line_hdls = plt.stem(freq, np.abs(xx_fft))
# plt.title('FFT')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Amplitud')
# axes_hdl = plt.gca()
# plt.show()

# plt.figure(3)
# line_hdls = plt.stem(freq, np.angle(xx_fft))
# plt.title('Fase')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Amplitud')
# axes_hdl = plt.gca()
# plt.show()

#%% Senoidal

vmax = 1
dc = 0
ff = 1
ph = 0
nn = 1000
fs = 1000 

tt,xx = senoidal(vmax, dc, ff, ph, nn, fs)

#%% Cuantización

Vref = 2
cont = 0

fig, axs = plt.subplots(3)

for B in [4,8,16]:
    #error de cuantizacion
    q = Vref/2**(B-1)
    x = np.round(xx/q)
    xq = x*q
    error = xq - xx

    axs[cont].plot(tt, xx,'b')
    axs[cont].plot(tt,xq,'r')
    axs[cont].plot(tt,error,'g')
    axs[cont].set_title("Vref = 2V y {} Bits".format(B))
    axs[cont].legend(['senoidal','cuantizada','error'])
    axs[cont].grid()
    axs[cont].set(xlabel='tiempo [s]', ylabel='Amplitud [V]')
    axs[cont].set_xlim(0,1)
    cont +=1

plt.show()

#%% Cuantización con ruido

ruido = np.random.normal(0,0.05, size=nn)
x_ruido = xx + ruido

Vref = 2
cont = 0

fig, axs = plt.subplots(3)

for B in [4,8,16]:
    #error de cuantizacion
    q = Vref/2**(B-1)
    x = np.round(x_ruido/q)
    xq = x*q
    error = xq - x_ruido

    axs[cont].plot(tt, x_ruido,'b')
    axs[cont].plot(tt,xq,'r')
    axs[cont].plot(tt,error,'g')
    axs[cont].set_title("Senoidal con ruido agregado, Vref = 2V y {} Bits".format(B))
    axs[cont].legend(['senoidal+ruido','cuantizada','error'])
    axs[cont].grid()
    axs[cont].set(xlabel='tiempo [s]', ylabel='Amplitud [V]')
    axs[cont].set_xlim(0,1)
    cont +=1

plt.show()


# plt.title('Cuantización con ruido agregado')
# plt.legend(['senoidal','cuantizada','error'])
# plt.grid()
# plt.show()

# bins = 10

# plt.figure(3)
# plt.hist(error_ruido, bins)

# plt.title('Histograma')
# plt.grid()
# plt.show()










