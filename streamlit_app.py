import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="KNN Trip Recommender", layout="wide")
st.title("⛵ KNN Trip Recommender - Kapal Listrik")
st.markdown("Sistem rekomendasi trip dengan K-Nearest Neighbor Algorithm")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/trip_kapal_listrik.csv')
        return df    except Exception as e:
        st.warning(f"Data tidak ditemukan: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Gagal memuat data")
    st.stop()

# Sidebar Summary
with st.sidebar:
    st.subheader("📊 Data Summary")
    st.metric("Total Trips", len(df))
    st.metric("Avg Passengers", f"{df['totalpenumpang'].mean():.1f}")
    st.metric("Max Capacity", df['kapasitaskursi'].max())

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data", "Charts", "KNN", "Export", "Info"])

# ===== TAB 1: DATA EXPLORER =====
with tab1:
    st.subheader("Trip Data Explorer")
    
    col1, col2 = st.columns(2)
    with col1:
        type_filter = st.multiselect(
            "Filter by Trip Type",
            df['typettrip'].unique(),
            default=df['typettrip'].unique()
        )
    with col2:
        day_filter = st.multiselect(
            "Filter by Day Type",
            df['jenishari'].unique(),
            default=df['jenishari'].unique()
        )
    
    # Apply filters
    filtered_df = df[
        (df['typettrip'].isin(type_filter)) & 
        (df['jenishari'].isin(day_filter))
    ].copy()
    
    st.write(f"Showing {len(filtered_df)} of {len(df)} trips")
    
    # Responsive data table
    st.dataframe(
        filtered_df[['idtrip', 'tanggal', 'jamberangkat', 'jamtiba', 'totalpenumpang', 
                     'kapasitaskursi', 'typettrip', 'jenishari', 'persentaseisi']].style.format({
            'persentaseisi': '{:.2f}',
            'totalpenumpang': '{:.0f}'
        }),
        use_container_width=True,
        height=400
    )

# ===== TAB 2: VISUALIZATIONS =====
with tab2:
    st.subheader("Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Passenger distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['totalpenumpang'], bins=15, color='skyblue', edgecolor='black')
        ax.set_xlabel('Total Passengers')
        ax.set_ylabel('Frequency')
        ax.set_title('Passenger Distribution')
        st.pyplot(fig)
    
    with col2:
        # Trip type distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        trip_counts = df['typettrip'].value_counts()
        colors = ['#FF6B6B' if x == 'Privat' else '#4ECDC4' for x in trip_counts.index]
        ax.pie(trip_counts.values, labels=trip_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Trip Type Distribution')
        st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Day type distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        day_counts = df['jenishari'].value_counts()
        ax.bar(day_counts.index, day_counts.values, color=['#95E1D3', '#F38181'])
        ax.set_xlabel('Day Type')
        ax.set_ylabel('Count')
        ax.set_title('Weekday vs Weekend Distribution')
        st.pyplot(fig)
    
    with col2:
        # Capacity utilization
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['persentaseisi'], bins=20, color='#A8D8EA', edgecolor='black')
        ax.set_xlabel('Capacity Utilization (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Capacity Utilization Distribution')
        st.pyplot(fig)

# ===== TAB 3: KNN RECOMMENDATIONS =====
with tab3:
    st.subheader("KNN-Based Trip Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_passengers = st.slider(
            "Number of Passengers",
            min_value=1,
            max_value=int(df['totalpenumpang'].max()),
            value=5
        )
    
    with col2:
        input_type = st.selectbox(
            "Preferred Trip Type",
            df['typettrip'].unique()
        )
    
    with col3:
        input_day = st.selectbox(
            "Day Type",
            df['jenishari'].unique()
        )
    
    # Simple KNN recommendation
    if st.button("Get Recommendations"):
        # Create feature matrix
        feature_cols = ['totalpenumpang', 'kapasitaskursi', 'persentaseisi']
        X = df[feature_cols].values
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Query point
        query = np.array([[input_passengers, 15, input_passengers/15]]).astype(float)
        query_scaled = scaler.transform(query)
        
        # Calculate distances
        distances = np.sqrt(np.sum((X_scaled - query_scaled[0])**2, axis=1))
        
        # Get top 5
        top_indices = np.argsort(distances)[:5]
        recommendations = df.iloc[top_indices][['idtrip', 'tanggal', 'jamberangkat', 'jamtiba', 
                                                 'totalpenumpang', 'typettrip', 'persentaseisi']].copy()
        recommendations['similarity_score'] = (1 / (1 + distances[top_indices])).round(3)
        
        st.write("**Top 5 Recommended Trips:**")
        st.dataframe(recommendations, use_container_width=True)

# ===== TAB 4: DATA EXPORT =====
with tab4:
    st.subheader("Export Trip Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_type = st.selectbox(
            "Select Trip Type to Export",
            ["All"] + list(df['typettrip'].unique())
        )
    
    with col2:
        export_day = st.selectbox(
            "Select Day Type to Export",
            ["All"] + list(df['jenishari'].unique())
        )
    
    # Filter for export
    export_df = df.copy()
    if export_type != "All":
        export_df = export_df[export_df['typettrip'] == export_type]
    if export_day != "All":
        export_df = export_df[export_df['jenishari'] == export_day]
    
    st.write(f"Exporting {len(export_df)} records")
    
    # CSV download
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

# ===== TAB 5: INFO =====
with tab5:
    st.subheader("📋 About This App")
    st.markdown("""
    **KNN Trip Recommender System** untuk sistem rekomendasi trip kapal listrik di Pantai Marina Boom.
    
    ### Features:
    - 📊 **Data Explorer**: Browse semua 87 trip records dengan filtering
    - 📈 **Visualizations**: Charts untuk distribusi data
    - 🤖 **KNN Recommendations**: Dapatkan trip recommendations berdasarkan preferensi Anda
    - 💾 **Data Export**: Download trip data dalam format CSV
    
    ### Dataset Info:
    - **Total Trips**: 87 records
    - **Trip Types**: Open Trip & Privat
    - **Features**: Date, Time, Passengers, Capacity, etc.
    
    ### Technology:
    - **Framework**: Streamlit
    - **Algorithm**: K-Nearest Neighbor (KNN)
    - **Data Processing**: Pandas, NumPy, Scikit-learn
    """)
    
    st.divider()
    st.write("**Dataset Overview:**")
    st.write(f"Total Records: {len(df)}")
    st.write(f"Columns: {', '.join(df.columns)}")
    st.dataframe(df.describe(), use_container_width=True)
