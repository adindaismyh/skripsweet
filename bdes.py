import  pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.metrics import mean_squared_error

def formatrupiah(uang):
    y = str(uang)
    if len(y) <= 3 :
        return 'Rp ' + y     
    else :
        p = y[-3:]
        q = y[:-3]
        return   formatrupiah(q) + '.' + p
        print ('Rp ' +  formatrupiah(q) + '.' + p )


def bdesmodel(data, datatraining,datatesting, alpha, panjang):
  s1 = np.zeros(len(data))
  s2 = np.zeros(len(data))
  prediksi = np.zeros(len(datatraining))
  slope = np.zeros(len(data))
  konstanta= np.zeros(len(data))
  prediksi[0] = datatraining['Harga'][0]
  for i in range(len(data)):
    if i < len(datatraining):
      if i < 1:
        s1[i] = datatraining['Harga'][i]
        s2[i]= datatraining['Harga'][i]
        konstanta[i] = 2*s1[i] - s2[i]
        slope[i] = (alpha/(1-alpha)) * (s1[i] - s2[i])

      else:
        s1[i] = (alpha * datatraining['Harga'][i]) +( (1-alpha) * s1[i-1])
        s2[i] = (alpha * s1[i] )+ ((1-alpha) * s2[i-1])
        konstanta[i] = (2*s1[i]) - s2[i]
        slope[i] = (alpha/(1-alpha)) * (s1[i] - s2[i])
        prediksi[i]= round(konstanta[i-1] + slope[i-1]*1)

    else:
        s1[i] = (alpha * datatesting['Harga'][i]) +( (1-alpha) * s1[i-1])
        s2[i] = (alpha * s1[i] )+ ((1-alpha) * s2[i-1])
        konstanta[i] = (2*s1[i]) - s2[i]
        slope[i] = (alpha/(1-alpha)) * (s1[i] - s2[i])

  hasiltesting = testing(data, datatraining, datatesting, konstanta, slope, panjang)
  return prediksi, hasiltesting, s1, s2, slope, konstanta


def testing(data, datatraining,datatesting, konstanta, slope, panjang_prediksi):
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



def golden_section(data,datatraining,datatesting,panjang, a, d,tol=0.00001):
  iter = 50
  r = (-1 +math.sqrt(5)) /2
  b = r * a + (1 - r) *d
  c = a+d - b
  k = 0

  while ((abs(d-a) > tol) and k < iter ) :
    k = k+1
    fb = bdesmodel(data,datatraining, datatesting, b, panjang)
    mapet_fb = round(mape(datatesting['Harga'], fb[1]), 5)
    fc = bdesmodel(data, datatraining, datatesting, c, panjang)
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

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

