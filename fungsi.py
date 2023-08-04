import streamlit as st
from PIL import Image
import plotly.graph_objs as go
import statistics
from bdes import formatrupiah
from sklearn.metrics import mean_squared_error
import numpy as np
from bdes import mape

def convert_df(df):
    return df.to_csv().encode('utf-8')

def foto(path, ukuran):
    foto=Image.open(path)
    foto=foto.resize((ukuran, ukuran))
    st.image(foto)

def foto2(path, ukuran1, ukuran2):
    foto=Image.open(path)
    foto=foto.resize((ukuran1, ukuran2))
    st.image(foto)

def kosumsi(path,tahun,jumlah,pertumbuhan,warna):
    img1=Image.open(path)
    img1=img1.resize((50,50))
    st.image(img1)
    st.markdown(f"<h1 style='text-align: right; padding-bottom:10px; font-size:20px; padding-top:0px'>TAHUN {tahun}</h1>", unsafe_allow_html=True) 
    st.markdown(f"<h1 style='text-align: center; padding-bottom:5px; font-size:27px; padding-top:0px; color:#6A51BC'>{jumlah}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: left; padding-bottom:5px; font-size:15px; padding-top:0px; color:{warna}'>{pertumbuhan}</h1>", unsafe_allow_html=True)

def visualisasitest(data, data2, data3, ukuran):
    data_1b = go.Scatter(x=data['Tanggal'], y=data['Harga'], name="Data Aktual", mode="lines")
    data_2b = go.Scatter(x=data2['Tanggal'], y=data2['prediksi'], name="Data Training", mode="lines")
    data_3b= go.Scatter(x=data3['Tanggal'], y=data3['prediksi'], name="Data Testing", mode="lines")

    figtesting = go.Figure([data_1b, data_2b, data_3b]) 
    figtesting.update_layout(xaxis_title = 'Tanggal', yaxis_title = 'Harga', width = ukuran)
    figtesting.update_layout(margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
    st.write(figtesting, unsafe_allow_html=True)

def visualisasitest2(data,ukuran):
    data_1b = go.Scatter(x=data['Tanggal'], y=data['data_aktual'], name="Data Aktual", mode="lines")
    data_2b= go.Scatter(x=data['Tanggal'], y=data['prediksi'], name="Data Testing", mode="lines")

    figtesting = go.Figure([data_1b, data_2b]) 
    figtesting.update_layout(xaxis_title = 'Tanggal', yaxis_title = 'Harga', width = ukuran)
    figtesting.update_layout(margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
    st.write(figtesting, unsafe_allow_html=True)

def visualisasi_prediksi(data, data1, data2, data3):
    data_1 = go.Scatter(x=data['Tanggal'], y=data['Harga'], name="Data Aktual", mode="lines")
    data_2 = go.Scatter(x=data1['Tanggal'], y=data1['prediksi'], name="Data Training", mode="lines")
    data_3= go.Scatter(x=data2['Tanggal'], y=data2['prediksi'], name="Data Testing", mode="lines")
    data_4= go.Scatter(x=data3['Tanggal'], y=data3['hasilprediksi'], name="Periode Depan", mode="lines")

    figtesting = go.Figure([data_1, data_2, data_3, data_4])
    figtesting.update_layout(xaxis_title = 'Tanggal',yaxis_title = 'Harga',width = 590)
    figtesting.update_layout(margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
    st.write(figtesting, unsafe_allow_html=True)

def statistika(data):
    st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px'>Statistika Deskriptif</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: right; color: #DE4D86;padding-bottom:0px;padding-top:25px;font-size: 33px'>{formatrupiah(max(data))}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: right; padding-bottom:0px; font-size:18px; padding-top:0px'>Harga Tertinggi</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: right; color: #DE4D86;padding-bottom:0px;padding-top:10px;font-size: 33px'>{formatrupiah(min(data))}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: right; padding-bottom:0px; font-size:18px; padding-top:0px'>Harga Terendah</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: right; color: #DE4D86;padding-bottom:0px;padding-top:10px;font-size: 33px'>{formatrupiah(round(statistics.mean(data)))}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: right; padding-bottom:30px; font-size:18px; padding-top:0px'>Rata - rata Harga</h1>", unsafe_allow_html=True)


def nilairmse(data1, data2):
    mse = mean_squared_error(data1, data2)
    rmse = np.sqrt(mse)
    return rmse

def visualisasitest3(data, data2, data3, sliding, ukuran):
    temp = str(sliding)
    data_1b = go.Scatter(x=data['Tanggal'], y=data['data_aktual'], name="Data Aktual", mode="lines")
    data_2b = go.Scatter(x=data2['Tanggal'], y=data2["prediksi " + temp], name="Prediksi Metode B-des", mode="lines")
    data_3b= go.Scatter(x=data3['Tanggal'], y=data3["prediksi " + temp], name="Prediksi Metode B-wema", mode="lines")

    figtesting = go.Figure([data_1b, data_2b, data_3b]) 
    figtesting.update_layout(xaxis_title = 'Tanggal', yaxis_title = 'Harga', width = ukuran)
    figtesting.update_layout(margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
    st.write(figtesting, unsafe_allow_html=True)

def penentuanmetode (data1, data2, sliding):
    temp = str(sliding)
    
    p1,p2 = st.columns([3,5])
    with p1:
        st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 23px; padding-top:20px; padding-left:20px;padding-right:5px;'>Metode Brown's Double Exponential Smoothing</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:3px; font-size: 20px; padding-top:5px; padding-left:10px;padding-right:10px;color: #6A51BC'>🚨 MAPE TESTING : {round(mape(data1['data_aktual'], data1['prediksi ' + temp]),2)} %</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size: 20px; padding-top:3px; padding-left:10px;padding-right:10px;color: #6A51BC'>🚨 RMSE TESTING : {formatrupiah(round(nilairmse(data1['data_aktual'], data1['prediksi ' + temp])))}</h1>", unsafe_allow_html=True)
        st.divider()

        st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 23px; padding-top:0px; padding-left:20px;padding-right:5px;'>Metode Browns Weighted Exponential Moving Average</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:3px; font-size: 20px; padding-top:5px; padding-left:10px;padding-right:10px;color: #6A51BC'>🚨 MAPE TESTING : {round(mape(data2['data_aktual'], data2['prediksi ' + temp]),2)} %</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:30px; font-size: 20px; padding-top:3px; padding-left:10px;padding-right:10px;color: #6A51BC'>🚨 RMSE TESTING : {formatrupiah(round(nilairmse(data2['data_aktual'], data2['prediksi ' + temp])))}</h1>", unsafe_allow_html=True)

            
    with p2:
        st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 18px; padding-top:15px; padding-left:10px;padding-right:10px;color: #DE4D86'>Visualisasi Metode B-DES dan B-WEMA dengan Sliding Window Sebesar {temp}</h1>", unsafe_allow_html=True)
        visualisasitest3(data1, data1, data2,sliding, 710)