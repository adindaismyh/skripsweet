import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import streamlit as st
from bdes import mape

def testing2(data, datatraining,datatesting, konstanta, slope, panjang_prediksi):
  prediksi_test=[]
  temp = len(datatraining)
  perulangan = 0
  while(temp < len(data)):
    for i in range (panjang_prediksi):
      if temp < len(data):
        if perulangan == 0:
          prediksi_test.append(round( konstanta[len(datatraining)-1]+ slope[len(datatraining)-1] *(i+2)))

        elif perulangan > 0:
          prediksi_test.append(round( konstanta[tempakhir]+ slope[tempakhir] *(i+2)))

        temp = temp +1

    tempakhir = temp - 1
    perulangan = perulangan +1

  return prediksi_test

def bdes2(data, datatraining,datatesting, alpha, panjang, Bt):
  s1 = np.zeros(len(data))
  s2 = np.zeros(len(data))
  prediksi = np.zeros(len(datatraining))
  slope = np.zeros(len(data))
  konstanta= np.zeros(len(data))
  prediksi[0] = datatraining['Harga'][0]

  for i in range(len(data)):
    if i < len(datatraining):
      if  i < 22:
        Bt[i] = datatraining['Harga'][i]

      s1[i] = (alpha * datatraining['Harga'][i]) +( (1-alpha) * Bt[i])
      s2[i] = (alpha * s1[i] )+ ((1-alpha) * Bt[i])
      konstanta[i] = (2*s1[i]) - s2[i]
      slope[i] = (alpha/(1-alpha)) * (s1[i] - s2[i])

      if i >0:
        prediksi[i]= round(konstanta[i-1] + slope[i-1]*1)
 

    else:
        s1[i] = (alpha * datatesting['Harga'][i]) +( (1-alpha) * Bt[i])
        s2[i] = (alpha * s1[i] )+ ((1-alpha) * Bt[i])
        konstanta[i] = (2*s1[i]) - s2[i]
        slope[i] = (alpha/(1-alpha)) * (s1[i] - s2[i])

  hasiltesting = testing2(data, datatraining, datatesting, konstanta, slope, panjang)
  return prediksi, hasiltesting, s1, s2, konstanta, slope

def golden_section2(data,datatraining,datatesting,panjang, Bt, a, d,tol=0.00001):
  iter = 50
  r = (-1 +math.sqrt(5)) /2
  k = 0
  b = r * a + (1 - r) *d
  c = a+d - b

  while ((abs(d-a) > tol) and (k < iter)):
    k  = k+1
    fb = bdes2(data,datatraining, datatesting, b, panjang, Bt)
    mapet_fb = round( mape(datatesting['Harga'], fb[1]),5)
    fc = bdes2(data, datatraining, datatesting, c, panjang, Bt)
    mapet_fc = round(mape(datatesting['Harga'], fc[1]),5)
    if (  (mapet_fb < mapet_fc)):
      d = c
      c = b
      b = r * a + (1 - r) * d
    else:
      a = b
      b = c
      c = a+ d - b


  return  round((d + a) / 2,5)

def nilaiBt (weights, data) :
  Bt = []
  n = len(weights)
  temp = 0
  for  i  in range (len(data)):
    if i == n:
      for j in range(len(weights)):
        temp = temp +(weights[j] * data['Harga'][n - j - 1])
      Bt.append(temp/sum(weights))
      n = n + 1
    else:
      Bt.append(0)
    temp = 0
  return Bt
