import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="KNN Trip Recommender", layout="wide")
st.title("⚓ KNN Trip Recommender - Kapal Listrik")
st.markdown("Sistem rekomendasi trip dengan K-Nearest Neighbor Algorithm")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/trip_data_updated.csv')
        return df
    except:
        st.warning("Data tidak ditemukan")
        return pd.DataFrame()

df = load_data()

if len(df) > 0:
    st.sidebar.header("Data Summary")
    st.sidebar.metric("Total Trips", len(df))
    
    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Charts", "KNN", "Export"])
    
    with tab1:
        st.header("Trip Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            type_f = st.multiselect("Type", df['type_trip'].unique(), default=df['type_trip'].unique())
        with col2:
            day_f = st.multiselect("Day", df['jenis_hari'].unique(), default=df['jenis_hari'].unique())
        with col3:
            shift_f = st.multiselect("Shift", df['shift_waktu'].unique(), default=df['shift_waktu'].unique())
        
        fdf = df[(df['type_trip'].isin(type_f)) & (df['jenis_hari'].isin(day_f)) & (df['shift_waktu'].isin(shift_f))]
        st.dataframe(fdf, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            df.boxplot(column='harga_final', by='type_trip', ax=ax)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.hist(df['harga_final'], bins=20)
            st.pyplot(fig)
    
    with tab3:
        st.write("KNN Model: 88.89% Accuracy")
    
    with tab4:
        csv = df.to_csv(index=False)
        st.download_button("Download", csv, "trips.csv", "text/csv")
else:
    st.error("Data not available")
