import streamlit as st
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statistics
from bdes import bdesmodel, mape, golden_section, formatrupiah
from bwema import golden_section2, nilaiBt,bdes2
from fungsi import convert_df, foto,kosumsi, visualisasitest2, nilairmse, penentuanmetode
import requests
from streamlit_lottie import st_lottie
from annotated_text import annotated_text, annotation



def beras():
    with open("style.css") as gaya:
        st.markdown(f'<style>{gaya.read()}</style>', unsafe_allow_html=True)


    col1,col2 = st.columns([4,2])
    with col1:
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:40px; padding-top:0px'>Hello, Everyone</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:18px; padding-top:10px'>Halaman dashboard komoditi beras </h1>", unsafe_allow_html=True)
        
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:45px'>Dataset</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:0px'>Berikut merupakan data set yang digunakan untuk membuat dan melatih model forecasting. Data yang digunakan merupakan data harga beras  mulai tanggal 1 Januari 2020 hingga 31 Juni 2023</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:13px; padding_top: 0px;color: #DE4D86'>Sumber : Web resmi PHPS Nasional bi.go.id</h1>", unsafe_allow_html=True)
        st.divider()
        beras = pd.read_csv("DATA/DATA BERAS.csv")
        beras = beras.assign(index = np.arange(len(beras)))
        beras['Tanggal'] = pd.to_datetime(beras['Tanggal'], infer_datetime_format=True)
        d1,d2 = st.columns([2,3])
        with d1:
            mindate = beras['Tanggal'][0]
            maxdate= beras['Tanggal'][867]
            st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Pilih tanggal untuk melihat data peramalan sesuai tanggal yang diinginkan</h1>", unsafe_allow_html=True)
            start_date = st.date_input(label="Mulai Tanggal",min_value=mindate,value=mindate)
            st.write(start_date)
            end_date = st.date_input(label="Hingga Tanggal",max_value=maxdate,value=maxdate)
            st.write(end_date)
            start_date = pd.to_datetime(start_date, infer_datetime_format=True)
            end_date = pd.to_datetime(end_date, infer_datetime_format=True)
                
            hasil=beras[(beras['Tanggal']>=start_date) &(beras['Tanggal']<=end_date) ]
            
        with d2:
            st.dataframe(hasil, use_container_width=True)
        
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:45px'>Visualisasi Data</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:0px'>Plot data harga beras  mulai bulan januari 2020 hingga maret 2023</h1>", unsafe_allow_html=True)
        
        v1,v2= st.columns([4,2])
        with v1:
            st.markdown(f"<h1 style='padding-top:10px'></h1>", unsafe_allow_html=True)
            tab1, tab2= st.tabs(["Diagram Line","Analisa"])
            with tab1:
                fig = px.line(hasil, x="Tanggal", y="Harga") 
                fig.update_layout(width=500,height=300,margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
                st.write(fig)
            with tab2:
                st.markdown(f"""<h1 style='text-align: justify; padding-right:25px; font-size:20px; padding-top:10px;padding-bottom:0px'>Berdasarkan gambar tersebut dapat disimpulkan bahwa pada periode awal harga beras mempunyai kecenderungan trend turun yang lambat kemudian di pertengahan tahun 2022 mengalami penurunan harga secara ekstrim dibandingkan bulan – bulan sebelumnya. 
                Penurunan ekstrim harga tersebut tidak berlangsung lama karena di akhir periode harga mengalami kecenderungan trend naik. </h1>""", unsafe_allow_html=True)

        with v2:
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px'>Statistika Deskriptif</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: right; color: #DE4D86;padding-bottom:0px;padding-top:25px;font-size: 33px'>{formatrupiah(max(hasil['Harga']))}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: right; padding-bottom:0px; font-size:18px; padding-top:0px'>Harga Tertinggi</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: right; color: #DE4D86;padding-bottom:0px;padding-top:10px;font-size: 33px'>{formatrupiah(min(hasil['Harga']))}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: right; padding-bottom:0px; font-size:18px; padding-top:0px'>Harga Terendah</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: right; color: #DE4D86;padding-bottom:0px;padding-top:10px;font-size: 33px'>{formatrupiah(round(statistics.mean(hasil['Harga'])))}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: right; padding-bottom:30px; font-size:18px; padding-top:0px'>Rata - rata Harga</h1>", unsafe_allow_html=True)
        
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px'>Metode Forecasting Kommoditi Sembako</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:13px; padding-top:8px; color: #674FBB'>Pilih metode di bawah ini untuk melihat hasil prediksi</h1>", unsafe_allow_html=True)
        st.divider()
        genre = st.radio( "OPTION", ('Brown’s Double Exponential Smoothing', 'Brown’s Weighted Exponential Moving Average', 'Perbandingan Metode'))
        st.markdown(f"<h1 style='text-align: left; padding-bottom:30px'></h1>", unsafe_allow_html=True)

    
    
    with col2:
        waktu = time.ctime()
        st.markdown(f"<h1 style='text-align: right; padding-bottom:30px; font-size:18px; padding-top:0px'>⏲️ {waktu}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:40px; padding-top:0px'>KOMODITI</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px'>Beras </h1>", unsafe_allow_html=True)
        d1,d2,d3 = st.columns(3)
        with d1:
            foto('gambar/prediksi.png',60)
            st.markdown(f"<h1 style='text-align: left; padding-bottom:15px; font-size:12px; padding-top:0px'>Prediksi</h1>", unsafe_allow_html=True)
        with d2:
            foto('gambar/analisis.png',60)
            st.markdown(f"<h1 style='text-align: left; padding-bottom:15px; font-size:12px; padding-top:0px'>Statistika</h1>", unsafe_allow_html=True)
        with d3:
            foto('gambar/plot.png',60)
            st.markdown(f"<h1 style='text-align: center; padding-bottom:15px; font-size:12px; padding-top:0px'>Visualisasi</h1>", unsafe_allow_html=True)

        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:20px; padding-top:0px'>Rata-Rata Konsumsi per Kapita beras (kg/kapita/tahun)</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:13px; padding-top:8px; color: #674FBB'>Sumber : Statistik Konsumsi Pangan Tahun 2022</h1>", unsafe_allow_html=True)
  
        st.divider()
        col3,col4 = st.columns(2)
        with col3:
            kosumsi('gambar/gambar1.png',2019,'78.429 kg','⬇️2.212 kg', "red")
        with col4:
            kosumsi('gambar/gambar2.png',2020,'78.487 kg','⬆️58 kg', "green" )
        col5,col6 = st.columns(2)
        with col5:
            kosumsi('gambar/gambar3.png',2021,'81.518 kg','⬆️3.031 kg', "green" )
        with col6:
            kosumsi('gambar/gambar3.png',2022,'81.004 kg','⬇️514 kg', "red" )
        
        data = pd.DataFrame({
            'Tahun' :["2020","2021","2022"],
            'Produksi':[54649202.24 , 54415294.22,54748977.00 ]
        })
        fig_produksi = px.bar(data, x='Tahun', y='Produksi', color="Tahun")
        
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:20px; padding-top:40px'>Produksi Beras  di Indonesia (ton)</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:13px; padding-top:8px; color: #674FBB'>Sumber : Bps.go.id </h1>", unsafe_allow_html=True)
        st.divider()
        fig_produksi.update_layout(width=360,height=300,margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="#D8D9F9")
        st.write(fig_produksi)
    
    
    Datates = 174
    trainberas = beras.iloc[:-Datates]
    testberas = beras.iloc[-Datates:]
    

    ## PREDIKSI BDES

    if genre == 'Brown’s Double Exponential Smoothing':
        opt =  golden_section(beras,trainberas,testberas,1, 0, 1)
        prediksi = prediksi = bdesmodel(beras, trainberas, testberas, opt,1)
        training ={'Tanggal': trainberas['Tanggal'],'data_aktual': trainberas['Harga'],'prediksi': prediksi[0]}
        training= pd.DataFrame(training)
        training['Tanggal'] = pd.to_datetime(training['Tanggal'], infer_datetime_format=True)


        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; font-style: italic'>Brown's Double Exponential Smoothing</h1>", unsafe_allow_html=True)
        st.markdown("_Brown’s Double Exponential Smoothing_  (B-DES) adalah metode peramalan yang dapat digunakan untuk data yang mempunyai pola _trend_, dimana keakuratannya bergantung dengan nilai parameter pemulusan ∝ yang bisa memperbaiki trend")
        tab5, tab6= st.tabs(["Prediksi Data Training","Prediksi Data Testing"])
        with tab5:
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Tabel Pediksi Data Training</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Tabel Hasil prediksi data training mulai tanggal 1 Januari 2020 hingga 31 Oktober 2022 </h1>", unsafe_allow_html=True)

            d1,d2 = st.columns([2,3])
            with d1:
                mindate1 = training['Tanggal'][0]
                maxdate1= training['Tanggal'][693]
                st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Pilih tanggal untuk melihat data peramalan sesuai tanggal yang diinginkan</h1>", unsafe_allow_html=True)            
                start_date1 = st.date_input(label="Tanggal Awal",min_value=mindate1,value=mindate1)
                st.write(start_date1)
                end_date1 = st.date_input(label="Tanggal Akhir",max_value=maxdate1,value=maxdate1)
                st.write(end_date1, use_container_width=True)
                start_date1 = pd.to_datetime(start_date1, infer_datetime_format=True)
                end_date1 = pd.to_datetime(end_date1, infer_datetime_format=True)
                        
                hasiltraining=training[(training['Tanggal']>=start_date1) &(training['Tanggal']<=end_date1) ]
                    
            with d2:
                st.dataframe(hasiltraining, use_container_width=True)
            
            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px; color:#6A51BC'>STATISTIKA DESKRIPTIF</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left ; padding-bottom:20px; font-size:18px ; padding-top:0px; font-style:normal'>Berikut merupakan statistika deskriptif dari {start_date1} hingga {end_date1}</h1>", unsafe_allow_html=True)
            s1,s2,s3 = st.columns(3)
            with s1:  
                sd1,sd2 = st.columns([2,3])
                with sd1:
                    foto('gambar/gambar5.png', 100)
                with sd2:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Tertinggi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:0px; font-size:38px ; padding-top:0px'>{formatrupiah(round(max(hasiltraining['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s2:
                sd1,sd2 = st.columns([2,3])
                with sd1:
                    foto('gambar/gambar6.png', 100)
                with sd2:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Terendah</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(min(hasiltraining['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s3:
                sd1,sd2 = st.columns([2,3])
                with sd1:
                    foto('gambar/gambar7.png', 100)
                with sd2:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Rata - Rata Harga Prediksi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(statistics.mean(hasiltraining['prediksi'])))}</h1>", unsafe_allow_html=True)

            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Visualisasi Data</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Visualisasi hasil prediksi data training harga beras dari model B- DES yang telah dibuat mulai tanggal {start_date1} hingga tanggal {end_date1}</h1>", unsafe_allow_html=True)

            v3,v4 = st.columns([5,3])
            with v3:
    
                fig = px.line(hasiltraining, x="Tanggal", y=["data_aktual", "prediksi"]) 
                fig.update_layout(margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
                st.write(fig, unsafe_allow_html=True)
            with v4:
                st.markdown(f"<h1 style='text-align: left; padding-bottom:5px; font-size:28px; padding-top:50px; color:#6A51BC'>ANALISA PLOT</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:18px; padding-top:0px'>Plot Hasil Data Training 1 Januari 2020 hingga 31 Oktober 2022</h1>", unsafe_allow_html=True)
                st.divider()
                st.markdown(f"""<h3 style='text-align: justify; padding-right:25px; font-size:18px;padding-bottom:30px; font-family: helvetica'>Sumbu x merupakan tanggal, dan sumbu y merupakan harga beras. Grafik warna biru merupakan data aktual, dan grafik warna biru muda merupakan hasil prediksi
                Nilai fitted value dari data training yang ditampilkan pada plot tersebut dapat disimpulkan bahwa hasil prediksi data training mampu mengikuti pola data aktual secara keseluruhan dengan baik, karena pola data prediksi terlihat sama dengan pola data aktual.</h3>""", unsafe_allow_html=True)
        
        with tab6:
            st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px; color:#6A51BC'>Window Size</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:18px ; padding-top:0px; font-style:normal'>Pilih panjang periode peramalan untuk sekali peramalan</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left ; padding-bottom:20px; font-size:16px ; padding-top:0px; color : #6A51BC'>Contoh : jika pengguna memilih window size sebesar 5 maka dalam sekali peramalan akan memprediksi 5 hari ke depan sekaligus</h1>", unsafe_allow_html=True)

            sliding = st.slider('', 1, 100, 5)
            pred_testing = bdesmodel(beras, trainberas, testberas, opt,sliding)
            testing ={'Tanggal': testberas['Tanggal'],'data_aktual': testberas['Harga'],'prediksi':pred_testing[1]}
            testing= pd.DataFrame(testing)
            testing['Tanggal'] = pd.to_datetime(testing['Tanggal'], infer_datetime_format=True)
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Tabel Pediksi Data Testing</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Tabel Hasil prediksi data testing mulai tanggal 01 November 2022 hingga 31 Juni 2023 </h1>", unsafe_allow_html=True)

            d1,d2 = st.columns([2,3])
            with d1:
                mindate2 = testing['Tanggal'][694]
                maxdate2= testing['Tanggal'][867]
                st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Pilih tanggal untuk melihat data peramalan sesuai tanggal yang diinginkan</h1>", unsafe_allow_html=True)            
                start_date2 = st.date_input(label="Pilih Tanggal Awal",min_value=mindate2,value=mindate2)
                st.write(start_date2)
                end_date2 = st.date_input(label="Pilih Tanggal Akhir",max_value=maxdate2,value=maxdate2)
                st.write(end_date2, use_container_width=True)
                start_date2 = pd.to_datetime(start_date2, infer_datetime_format=True)
                end_date2 = pd.to_datetime(end_date2, infer_datetime_format=True)   
                        
                hasiltesting=testing[(testing['Tanggal']>=start_date2) &(testing['Tanggal']<=end_date2) ]
                    
            with d2:
                st.dataframe(hasiltesting, use_container_width=True)
            
            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px; color:#6A51BC'>STATISTIKA DESKRIPTIF</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left ; padding-bottom:20px; font-size:18px ; padding-top:0px; font-style:normal'>Berikut merupakan statistika deskriptif dari {start_date2} hingga {end_date2}</h1>", unsafe_allow_html=True)
        
            s1,s2,s3 = st.columns(3)
            with s1:  
                sd1,sd2 = st.columns([2,3])
                with sd1:
                    foto('gambar/gambar5.png', 100)
                with sd2:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Tertinggi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:0px; font-size:38px ; padding-top:0px'>{formatrupiah(round(max(hasiltesting['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s2:
                sd1,sd2 = st.columns([2,3])
                with sd1:
                    foto('gambar/gambar6.png', 100)
                with sd2:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Terendah</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(min(hasiltesting['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s3:
                sd1,sd2 = st.columns([2,3])
                with sd1:
                    foto('gambar/gambar7.png', 100)
                with sd2:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Rata - Rata Harga Prediksi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(statistics.mean(hasiltesting['prediksi'])))}</h1>", unsafe_allow_html=True)

            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Visualisasi Data</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Visualisasi hasil prediksi data testing harga beras dari model B- DES yang telah dibuat mulai tanggal {start_date2} hingga tanggal {end_date2}</h1>", unsafe_allow_html=True)
            v3, v4, v5 =st.columns([4,2,1])
            with v3:
                visualisasitest2(testing, 700)
                
            with v4:
                st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size:26px; color:#6A51BC'>EVALUASI PERAMALAN</h1>", unsafe_allow_html=True)
                st.info('MAPE MODEL',icon="📌")
                st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:5px; font-size:45px'>{round(mape(training['data_aktual'], training['prediksi']),2)} %</h1>", unsafe_allow_html=True)
                st.info('MAPE TESTING',icon="📌")
                st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:5px; font-size:45px'; padding-top:0px>{round(mape(testing['data_aktual'], testing['prediksi']),2)} %</h1>", unsafe_allow_html=True)
                st.info('RMSE TESTING',icon="📌")
                st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:20px; font-size:45px'> {formatrupiah(round(nilairmse(testing['data_aktual'], testing['prediksi'])))}</h1>", unsafe_allow_html=True)


    # PREDIKSI BWEMA 

    

    if genre == 'Brown’s Weighted Exponential Moving Average':
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:45px; font-style: italic'>Brown's Weighted Exponential Moving Average</h1>", unsafe_allow_html=True)
        st.markdown("_Brown’s Weighted Exponential Moving Average_ adalah gabungan antara metode WMA (_Weighted Moving Average_) dan B-DES dimana metode tersebut dapat digunakan untuk meramalkan data deret waktu masa depan yang mempunyai pola trend ")

        weights =[22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
        Bt_m = nilaiBt(weights, beras)
        
        opt2 =  golden_section2(beras,trainberas,testberas,1,Bt_m, 0, 1)
        prediksi2 = bdes2(beras, trainberas, testberas, opt2,1, Bt_m)
        training2 ={'Tanggal': trainberas['Tanggal'],'data_aktual': trainberas['Harga'],'prediksi': prediksi2[0]}
        training2= pd.DataFrame(training2)
        training2['Tanggal'] = pd.to_datetime(training2['Tanggal'], infer_datetime_format=True)
        tab5_b, tab6_b= st.tabs(["Prediksi Data Training","Prediksi Data Testing"])
        with tab5_b:
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Tabel Pediksi Data Training</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Tabel Hasil prediksi data training mulai tanggal 1 Januari 2020 hingga 31 Oktober 2022 </h1>", unsafe_allow_html=True)

            d1_b,d2_b = st.columns([2,3])
            with d1_b:
                mindate1_b = training2['Tanggal'][0]
                maxdate1_b= training2['Tanggal'][693]
                st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Pilih tanggal untuk melihat data peramalan sesuai tanggal yang diinginkan</h1>", unsafe_allow_html=True)            
                start_date1_b = st.date_input(label="Tanggal Awal :",min_value=mindate1_b,value=mindate1_b)
                st.write(start_date1_b)
                end_date1_b = st.date_input(label="Tanggal Akhir :",max_value=maxdate1_b,value=maxdate1_b)
                st.write(end_date1_b, use_container_width=True)
                start_date1_b = pd.to_datetime(start_date1_b, infer_datetime_format=True)
                end_date1_b = pd.to_datetime(end_date1_b, infer_datetime_format=True)
                        
                hasiltraining2=training2[(training2['Tanggal']>=start_date1_b) &(training2['Tanggal']<=end_date1_b) ]
                    
            with d2_b:
                st.dataframe(training2, use_container_width=True)
            
            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px; color:#6A51BC'>STATISTIKA DESKRIPTIF</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left ; padding-bottom:20px; font-size:18px ; padding-top:0px; font-style:normal'>Berikut merupakan statistika deskriptif dari {start_date1_b} hingga {end_date1_b}</h1>", unsafe_allow_html=True)
            s1_b,s2_b,s3_b = st.columns(3)
            with s1_b:  
                sd1_b,sd2_b = st.columns([2,3])
                with sd1_b:
                    foto('gambar/gambar5.png', 100)
                with sd2_b:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Tertinggi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:0px; font-size:38px ; padding-top:0px'>{formatrupiah(round(max(hasiltraining2['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s2_b:
                sd1_b,sd2_b = st.columns([2,3])
                with sd1_b:
                    foto('gambar/gambar6.png', 100)
                with sd2_b:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Terendah</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(min(hasiltraining2['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s3_b:
                sd1_b,sd2_b = st.columns([2,3])
                with sd1_b:
                    foto('gambar/gambar7.png', 100)
                with sd2_b:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Rata - Rata Harga Prediksi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(statistics.mean(hasiltraining2['prediksi'])))}</h1>", unsafe_allow_html=True)

            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Visualisasi Data</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Visualisasi hasil prediksi data training harga beras dari model B- DES yang telah dibuat mulai tanggal {start_date1_b} hingga tanggal {end_date1_b}</h1>", unsafe_allow_html=True)

            v3_b,v4_b = st.columns([5,3])
            with v3_b:
    
                fig2 = px.line(hasiltraining2, x="Tanggal", y=["data_aktual", "prediksi"]) 
                fig2.update_layout(margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
                st.write(fig2, unsafe_allow_html=True)
            with v4_b:
                st.markdown(f"<h1 style='text-align: left; padding-bottom:5px; font-size:28px; padding-top:50px; color:#6A51BC'>ANALISA PLOT</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:18px; padding-top:0px'>Plot Hasil Data Training 1 Januari 2020 hingga 31 Oktober 2022</h1>", unsafe_allow_html=True)
                st.divider()
                st.markdown(f"""<h3 style='text-align: justify; padding-right:25px; font-size:18px;padding-bottom:30px; font-family: helvetica'>Sumbu x merupakan tanggal, dan sumbu y merupakan harga beras. Grafik warna biru merupakan data aktual, dan grafik warna biru muda merupakan hasil prediksi
                Nilai fitted value dari data training yang ditampilkan pada plot tersebut dapat disimpulkan bahwa hasil prediksi mampu mengikuti pola data aktual secara keseluruhan dengan baik, karena pola data prediksi terlihat sama dengan pola data aktual.</h3>""", unsafe_allow_html=True)
       
        with tab6_b:
            st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px; color:#6A51BC'>Window Size</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:18px ; padding-top:0px; font-style:normal'>Pilih panjang periode peramalan untuk sekali peramalan</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left ; padding-bottom:20px; font-size:16px ; padding-top:0px; color : #6A51BC'>Contoh : jika pengguna memilih window size sebesar 5 maka dalam sekali peramalan akan memprediksi 5 hari ke depan sekaligus</h1>", unsafe_allow_html=True)

            sliding2 = st.slider('', 1, 100, 5)
            pred_testing2 = bdes2(beras, trainberas, testberas, opt2,sliding2, Bt_m)
            testing2 ={'Tanggal': testberas['Tanggal'],'data_aktual': testberas['Harga'],'prediksi':pred_testing2[1], 'nilaidasar' : Bt_m[-Datates:]}
            testing2= pd.DataFrame(testing2)
            testing2['Tanggal'] = pd.to_datetime(testing2['Tanggal'], infer_datetime_format=True)
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Tabel Pediksi Data Testing</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Tabel Hasil prediksi data testing mulai tanggal 19 Desember 2022 hingga 31 Mei 2023 </h1>", unsafe_allow_html=True)

            d1_b,d2_b = st.columns([2,3])
            with d1_b:
                mindate2_b = testing2['Tanggal'][694]
                maxdate2_b= testing2['Tanggal'][867]
                st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Pilih tanggal untuk melihat data peramalan sesuai tanggal yang diinginkan</h1>", unsafe_allow_html=True)            
                start_date2_b = st.date_input(label="Pilih Tanggal Awal :",min_value=mindate2_b,value=mindate2_b)
                st.write(start_date2_b)
                end_date2_b = st.date_input(label="Pilih Tanggal Akhir :",max_value=maxdate2_b,value=maxdate2_b)
                st.write(end_date2_b, use_container_width=True)
                start_date2_b = pd.to_datetime(start_date2_b, infer_datetime_format=True)
                end_date2_b = pd.to_datetime(end_date2_b, infer_datetime_format=True)   
                        
                hasiltesting2=testing2[(testing2['Tanggal']>=start_date2_b) &(testing2['Tanggal']<=end_date2_b) ]
                csv2 = convert_df(hasiltesting2)
                st.markdown(f"<h1 style='text-align: left; color: #6C5CD7 ; padding-bottom:10px; font-size:15px ; padding-top:0px'>Simpan data hasil prediksi</h1>", unsafe_allow_html=True)
                st.download_button(
                    label="Download data as CSV",
                    data= csv2,
                    file_name='berasgoreng_bwema.csv',
                    mime='text/csv',
                )
                st.markdown(f"<h1 style='padding-bottom:15px'></h1>", unsafe_allow_html=True)
                    
            with d2_b:
                st.dataframe(hasiltesting2, use_container_width=True)
            
            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px; color:#6A51BC'>STATISTIKA DESKRIPTIF</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left ; padding-bottom:20px; font-size:18px ; padding-top:0px; font-style:normal'>Berikut merupakan statistika deskriptif dari {start_date2_b} hingga {end_date2_b}</h1>", unsafe_allow_html=True)
        
            s1_b,s2_b,s3_b = st.columns(3)
            with s1_b:  
                sd1_b,sd2_b = st.columns([2,3])
                with sd1_b:
                    foto('gambar/gambar5.png', 100)
                with sd2_b:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Tertinggi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:0px; font-size:38px ; padding-top:0px'>{formatrupiah(round(max(hasiltesting2['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s2_b:
                sd1_b,sd2_b = st.columns([2,3])
                with sd1_b:
                    foto('gambar/gambar6.png', 100)
                with sd2_b:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Terendah</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(min(hasiltesting2['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s3_b:
                sd1_b,sd2_b = st.columns([2,3])
                with sd1_b:
                    foto('gambar/gambar7.png', 100)
                with sd2_b:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Rata - Rata Harga Prediksi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(statistics.mean(hasiltesting2['prediksi'])))}</h1>", unsafe_allow_html=True)

            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Visualisasi Data</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Visualisasi hasil prediksi data testing harga beras dari model B- DES yang telah dibuat mulai tanggal {start_date2_b} hingga tanggal {end_date2_b}</h1>", unsafe_allow_html=True)
            v3_b, v4_b, v5_b =st.columns([4,2,1])
            with v3_b:
                visualisasitest2(testing2, 670)
                
            with v4_b:
                st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size:26px; color:#6A51BC'>EVALUASI PERAMALAN</h1>", unsafe_allow_html=True)
                st.info('MAPE MODEL',icon="📌")
                st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:5px; font-size:45px'>{round(mape(training2['data_aktual'], training2['prediksi']),2)} %</h1>", unsafe_allow_html=True)
                st.info('MAPE TESTING',icon="📌")
                st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:5px; font-size:45px'; padding-top:0px>{round(mape(testing2['data_aktual'], testing2['prediksi']),2)} %</h1>", unsafe_allow_html=True)
                st.info('RMSE TESTING',icon="📌")
                st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:20px; font-size:45px'> {formatrupiah(round(nilairmse(testing2['data_aktual'], testing2['prediksi'])))}</h1>", unsafe_allow_html=True)

    
    if genre == 'Perbandingan Metode' :
        st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px'>PERBANDINGAN ANTARA METODE B - DES DAN B - WEMA</h1>", unsafe_allow_html=True)
        st.markdown("Perbadingan ini didasarkan pada nilai kesalahan untuk menentukan rentang prediksi optimal dalam sekali prediksi dan menentukan metode mana yang lebih cocok untuk prediksi komoditi sembako")

        st.divider()
        #### B-DES
        opt =  golden_section(beras,trainberas,testberas,1, 0, 1)
        b_des ={'Tanggal': testberas['Tanggal'],'data_aktual': testberas['Harga']}
        b_des= pd.DataFrame(b_des)
        b_des['Tanggal'] = pd.to_datetime(b_des['Tanggal'], infer_datetime_format=True)
        sliding = [1,5,10,30,50,70]
        mape_bdes= []
        mape_bwema= []
        rmse_bdes = []
        rmse_bwema= []
        
        for i in range(len(sliding)):
            pred_testing = bdesmodel(beras, trainberas, testberas, opt,sliding[i])
            temp = str(sliding[i])
            b_des["prediksi " + temp] = pred_testing[1]
            mape_bdes.append(round(mape(b_des['data_aktual'], b_des["prediksi " + temp]),2))
            rmse_bdes.append(round(nilairmse(b_des['data_aktual'], b_des["prediksi " + temp])))
        

        ### B-WEMA
        weights =[22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
        Bt_m = nilaiBt(weights, beras)
        
        
        opt2 =  golden_section2(beras,trainberas,testberas,1,Bt_m, 0, 1)
        b_wema ={'Tanggal': testberas['Tanggal'],'data_aktual': testberas['Harga']}
        b_wema= pd.DataFrame(b_wema)
        b_wema['Tanggal'] = pd.to_datetime(b_wema['Tanggal'], infer_datetime_format=True)

        for i in range(len(sliding)):
            pred_testing = bdes2(beras, trainberas, testberas, opt2,sliding[i], Bt_m)
            temp = str(sliding[i])
            b_wema["prediksi " + temp] = pred_testing[1]
            mape_bwema.append(round(mape(b_wema['data_aktual'], b_wema["prediksi " + temp]),2))
            rmse_bwema.append(round(nilairmse(b_wema['data_aktual'], b_wema["prediksi " + temp])))

        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size: 30px; padding-top:20px; color:#6A51BC'>SLIDING WINDOW </h1>", unsafe_allow_html=True)
        st.markdown("Berikut merupakan hasil evaluasi MAPE dan RMSE serta visualisasi dari metode B-DES dan B-WEMA dengan sliding window")

        for i in range(len(sliding)):
            temp = str(sliding[i])
            with st.expander("Hasil Sliding Window " + temp):
                penentuanmetode (b_des, b_wema, sliding[i])
        
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size: 30px; padding-top:20px; color:#6A51BC'>METODE TERBAIK</h1>", unsafe_allow_html=True)
        st.markdown("Menampilkan optimal panjang periode sekali peramalan dan metode terbaik dalam prediksi komoditi beras")


        col1,col2= st.columns([2,7])
        with col1:
            url = requests.get("https://lottie.host/03758801-ecd0-4b77-af61-44fdecb1f81d/wVkF9qzeAb.json")
            url_json = dict()
            if url.status_code == 200:
                url_json = url.json()
            else:
                print("Error in URL")
            
            st_lottie(url_json,
                    reverse=True,height=300,  width=300, speed=1,  loop=True, quality='high',key = 'prediksi')
            
        
        with col2:
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size: 25px; padding-top:20px; color:#6A51BC'>Panjang Periode Peramalan</h1>", unsafe_allow_html=True)
            annotated_text("Prediksi komoditi beras menggunakan metode B-DES dan B-WEMA dalam sekali peramalan  ",annotation(" lebih optimal berada pada rentang","1 hingga 10 hari", "#faa", font_family="Comic Sans MS", font_weight="bold"),"  ke depan dengan  " ,annotation("selisih","di bawah Rp 300 ", "#faf", font_family="Comic Sans MS", font_weight="bold")," dari hasil peramalan data testing selama 174 hari. Sedangkan untuk  ", annotation("window size", "sebesar 30 hingga 70", font_family="Comic Sans MS", font_weight="bold"),"  tidak disarankan karena mempunyai  ", annotation("selisih"," lebih dari Rp 500", "#afa", font_family="Comic Sans MS", font_weight="bold"),"  dan berdasarkan visualisasi diagram line dapat dilihat bahwa dengan window size tersebut hasil peramalan tidak mampu mengikuti pola data aktual")
            st.markdown(f"<h1 style='text-align: left; padding-bottom:15px; font-size: 25px; padding-top:10px; color:#6A51BC'>Pemilihan Metode Peramalan Terbaik Komoditi beras</h1>", unsafe_allow_html=True)
            ratabdes= [round(statistics.mean(mape_bdes),2)]
            evaluasi = pd.DataFrame(ratabdes)
            evaluasi['rmse bdes (Rp)'] = round(statistics.mean(rmse_bdes))
            evaluasi['mape bwema (%)'] = round(statistics.mean(mape_bwema),2)
            evaluasi['rmse bwema (Rp)'] = round(statistics.mean(rmse_bwema))
            evaluasi = evaluasi.rename(columns = {0: "mape bdes (%)"})
            st.dataframe(evaluasi,  use_container_width=True)
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size: 20px; padding-top:0px; color:#6A51BC'>KETERANGAN</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size: 18px; padding-top:2px;'>Metode terbaik untuk memprediksi komoditi harga beras adalah Brown’s Double Exponential Smoothing (B-DES) karena menghasilkan rata – rata terkecil nilai MAPE sebesar {round(statistics.mean(mape_bdes),2)} % dan rmse sebesar {formatrupiah(round(statistics.mean(rmse_bdes)))}. </h1>", unsafe_allow_html=True)

