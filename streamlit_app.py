import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="KNN Trip Recommender", layout="wide")
st.title("⛵ KNN Trip Recommender - Kapal Listrik")
st.markdown("Sistem rekomendasi trip dengan K-Nearest Neighbor Algorithm")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/trip_kapal_listrik_data_rapi.csv', sep=';')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None or df.empty:
    st.error("Failed to load data. Please check if trip_kapal_listrik_data_rapi.csv exists in data folder.")
    st.stop()

# Sidebar Summary
with st.sidebar:
    st.subheader("📊 Data Summary")
    st.metric("Total Trips", len(df))
    try:
        st.metric("Avg Passengers", f"{df['total_penumpang'].mean():.1f}")
        st.metric("Max Capacity", int(df['kapasitas_kursi'].max()))
    except Exception as e:
        st.warning(f"Could not calculate metrics: {e}")

# Main Tabs

# TAB 1: DATA EXPLORER
with tab1:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data", "Charts", "KNN", "Export", "Info", "Model Analysis"])
    
    col1, col2 = st.columns(2)
    with col1:
        try:
            type_filter = st.multiselect(
                "Filter by Trip Type",
                df['type_trip'].unique() if 'type_trip' in df.columns else [],
                default=df['type_trip'].unique() if 'type_trip' in df.columns else []
            )
        except Exception as e:
            st.warning(f"Could not filter by trip type: {e}")
            type_filter = []
    
    with col2:
        try:
            day_filter = st.multiselect(
                "Filter by Day Type",
                df['jenishari'].unique() if 'jenishari' in df.columns else [],
                default=df['jenishari'].unique() if 'jenishari' in df.columns else []
            )
        except Exception as e:
            st.warning(f"Could not filter by day type: {e}")
            day_filter = []
    
    # Apply filters
    try:
        if type_filter and day_filter:
            filtered_df = df[
                (df['type_trip'].isin(type_filter)) & 
                (df['jenishari'].isin(day_filter))
            ].copy()
        else:
            filtered_df = df.copy()
        
        st.write(f"Showing {len(filtered_df)} of {len(df)} trips")
        
        # Display data
        display_cols = ['idtrip', 'tanggal', 'jamberangkat', 'jamtiba', 'total_penumpang', 
                        'kapasitas_kursi', 'type_trip', 'jenishari']
        available_cols = [c for c in display_cols if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_cols],
            use_container_width=True,
            height=400
        )
    except Exception as e:
        st.error(f"Error displaying data: {e}")

# TAB 2: VISUALIZATIONS
with tab2:
    st.subheader("Data Visualizations")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'total_penumpang' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(pd.to_numeric(df['total_penumpang'], errors='coerce'), bins=15, color='skyblue', edgecolor='black')
                ax.set_xlabel('Total Passengers')
                ax.set_ylabel('Frequency')
                ax.set_title('Passenger Distribution')
                st.pyplot(fig)
        
        with col2:
            if 'type_trip' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                trip_counts = df['type_trip'].value_counts()
                colors = ['#FF6B6B', '#4ECDC4']
                ax.pie(trip_counts.values, labels=trip_counts.index, autopct='%1.1f%%', colors=colors[:len(trip_counts)], startangle=90)
                ax.set_title('Trip Type Distribution')
                st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'jenishari' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                day_counts = df['jenishari'].value_counts()
                ax.bar(day_counts.index, day_counts.values, color=['#95E1D3', '#F38181'])
                ax.set_xlabel('Day Type')
                ax.set_ylabel('Count')
                ax.set_title('Weekday vs Weekend Distribution')
                st.pyplot(fig)
        
        with col2:
            if 'persentaseisi' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(pd.to_numeric(df['persentaseisi'], errors='coerce'), bins=20, color='#A8D8EA', edgecolor='black')
                ax.set_xlabel('Capacity Utilization (%)')
                ax.set_ylabel('Frequency')
                ax.set_title('Capacity Utilization Distribution')
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")

# TAB 3: KNN RECOMMENDATIONS
with tab3:
    st.subheader("KNN-Based Trip Recommendations")
    
    try:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_passengers = st.slider(
                "Number of Passengers",
                min_value=1,
                max_value=int(pd.to_numeric(df['total_penumpang'], errors='coerce').max()),
                value=5
            )
        
        with col2:
            if 'type_trip' in df.columns:
                input_type = st.selectbox(
                    "Preferred Trip Type",
                    df['type_trip'].unique()
                )
            else:
                input_type = "Open Trip"
        
        with col3:
            if 'jenishari' in df.columns:
                input_day = st.selectbox(
                    "Day Type",
                    df['jenishari'].unique()
                )
            else:
                input_day = "Weekday"
        
        if st.button("Get Recommendations"):
            feature_cols = ['total_penumpang', 'kapasitas_kursi']
            available_features = [c for c in feature_cols if c in df.columns]
            
            if len(available_features) >= 2:
                X = df[available_features].apply(pd.to_numeric, errors='coerce').values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                query = np.array([[input_passengers, 15]])
                query_scaled = scaler.transform(query)
                
                distances = np.sqrt(np.sum((X_scaled - query_scaled[0])**2, axis=1))
                top_indices = np.argsort(distances)[:5]
                
                rec_cols = ['idtrip', 'tanggal', 'jamberangkat', 'total_penumpang', 'type_trip']
                available_rec_cols = [c for c in rec_cols if c in df.columns]
                
                recommendations = df.iloc[top_indices][available_rec_cols].copy()
                recommendations['similarity_score'] = (1 / (1 + distances[top_indices])).round(3)
                
                st.write("**Top 5 Recommended Trips:**")
                st.dataframe(recommendations, use_container_width=True)
            else:
                st.error("Not enough features for KNN calculation")
    except Exception as e:
        st.error(f"Error in KNN recommendations: {e}")

# TAB 4: DATA EXPORT
with tab4:
    st.subheader("Export Trip Data")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            export_types = ["All"]
            if 'type_trip' in df.columns:
                export_types.extend(df['type_trip'].unique())
            export_type = st.selectbox("Select Trip Type to Export", export_types)
        
        with col2:
            export_days = ["All"]
            if 'jenishari' in df.columns:
                export_days.extend(df['jenishari'].unique())
            export_day = st.selectbox("Select Day Type to Export", export_days)
        
        export_df = df.copy()
        if export_type != "All" and 'type_trip' in df.columns:
            export_df = export_df[export_df['type_trip'] == export_type]
        if export_day != "All" and 'jenishari' in df.columns:
            export_df = export_df[export_df['jenishari'] == export_day]
        
        st.write(f"Exporting {len(export_df)} records")
        
        csv_buffer = BytesIO()
        export_df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        
        st.download_button(
            label="📥 Download as CSV",
            data=csv_buffer.getvalue(),
            file_name="trip_data_export.csv",
            mime="text/csv"
        )
        
        st.write("**Preview:**")
        st.dataframe(export_df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Error in export: {e}")

# TAB 5: INFO
with tab5:
    st.subheader("📋 About This App")
    st.markdown("""
    **KNN Trip Recommender System** untuk sistem rekomendasi trip kapal listrik di Pantai Marina Boom.
    
    ### Features:
    - 📊 **Data Explorer**: Browse semua trip records dengan filtering
    - 📈 **Visualizations**: Charts untuk distribusi data
    - 🤖 **KNN Recommendations**: Dapatkan trip recommendations berdasarkan preferensi
    - 💾 **Data Export**: Download trip data dalam format CSV
    
    ### Dataset Info:
    - **Total Trips**: """ + str(len(df)) + """ records
    - **Trip Types**: Open Trip & Privat
    - **Columns**: """ + ", ".join(df.columns) + """
    
    ### Technology:
    - **Framework**: Streamlit
    - **Algorithm**: K-Nearest Neighbor (KNN)
    - **Data Processing**: Pandas, NumPy, Scikit-learn
    """)
    
    st.divider()
    st.write("**Dataset Overview:**")
    st.write(f"Total Records: {len(df)}")
    st.dataframe(df.describe(), use_container_width=True)

# TAB 6: MODEL ANALYSIS
with tab6:
 st.subheader("🤖 KNN Model Documentation & Analysis")
 
 st.markdown("### Algoritma K-Nearest Neighbor (KNN)")
 st.write("""
 **KNN** adalah algoritma machine learning yang berbasis pada prinsip similarity.
 Algoritma ini menemukan K data points terdekat dan menggunakan mayoritas label mereka untuk prediksi.
 """)
 
 col1, col2 = st.columns(2)
 with col1:
     st.markdown("#### Parameter Model")
     st.write("""
     - **K (Neighbors)**: 5
     - **Distance Metric**: Euclidean
     - **Features Used**: 2 (total_penumpang, kapasitas_kursi)
     - **Data Points**: 87 trips
     """)
 
 with col2:
     st.markdown("#### Fitur yang Digunakan")
     st.write("""
     1. **Total Penumpang**: Jumlah penumpang dalam trip
     2. **Kapasitas Kursi**: Total kapasitas kapal
        
     Fitur ini di-normalize menggunakan StandardScaler
     untuk memastikan scale yang sama dalam perhitungan jarak.
     """)
 
 st.divider()
 st.markdown("### Analisis Trip Type")
 
 col1, col2 = st.columns(2)
 with col1:
     if 'type_trip' in df.columns:
     trip_dist = df['type_trip'].value_counts()
     st.write("**Distribusi Trip Type:**")
     st.bar_chart(trip_dist)
 
 with col2:
 if 'jenis_hari' in df.columns:
 day_dist = df['jenis_hari'].value_counts()
 st.write("**Distribusi Hari:**")
 st.pie_chart(day_dist)
 
 st.divider()
 st.markdown("### Cara Kerja Sistem Rekomendasi")
 st.write("""
 1. **Input User**: Jumlah penumpang, tipe trip, dan tipe hari
 2. **Normalisasi**: Data input di-normalize sesuai scaling data training
 3. **Hitung Jarak**: Hitung Euclidean distance ke semua trips dalam database
 4. **Ambil K Terdekat**: Pilih 5 trips dengan jarak terkecil
 5. **Tampilkan Hasil**: Urutkan berdasarkan similarity score
 """)
