import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load CSV data"""
    df = pd.read_csv(filepath)
    return df

def add_type_trip(df):
    """Add type_trip column (80% Open Trip, 20% Privat)"""
    np.random.seed(42)
    df['type_trip'] = np.random.choice(
        ['Open Trip', 'Privat'],
        size=len(df),
        p=[0.8, 0.2]
    )
    return df

def preprocess_features(df):
    """Extract and create new features"""
    df['jam_berangkat'] = pd.to_datetime(df['jam_berangkat'], format='%H:%M')
    df['jam_tiba'] = pd.to_datetime(df['jam_tiba'], format='%H:%M')
    
    df['durasi_menit'] = (df['jam_tiba'] - df['jam_berangkat']).dt.total_seconds() / 60
    df['persentase_isi'] = df['total_penumpang'] / df['kapasitas_kursi']
    
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df['hari'] = df['tanggal'].dt.day_name()
    df['jenis_hari'] = df['tanggal'].dt.weekday.apply(
        lambda x: 'Weekend' if x >= 5 else 'Weekday'
    )
    
    def shift_waktu(h):
        if 5 <= h < 11:
            return 'Pagi'
        elif 11 <= h < 15:
            return 'Siang'
        else:
            return 'Sore'
    
    df['shift_waktu'] = df['jam_berangkat'].dt.hour.apply(shift_waktu)
    df = df[df['keterangan_operasi'] == 'Normal'].copy()
    
    return df

def encode_features(df):
    """One-hot encode categorical features"""
    fitur_kategori = ['hari', 'jenis_hari', 'shift_waktu']
    df_encoded = pd.get_dummies(df, columns=fitur_kategori)
    
    df_encoded['jenis_trip'] = np.where(
        df_encoded['persentase_isi'] == 1.0,
        'Penuh',
        'Tidak Penuh'
    )
    
    return df_encoded

def get_feature_columns(df_encoded):
    """Get list of feature columns for model"""
    fitur_kategori = ['hari', 'jenis_hari', 'shift_waktu']
    feature_cols = [
        'durasi_menit',
        'total_penumpang',
        'kapasitas_kursi',
        'persentase_isi'
    ] + [col for col in df_encoded.columns if col.startswith(tuple(fitur_kategori))]
    
    return feature_cols

def scale_features(X, scaler=None):
    """Normalize features using StandardScaler"""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler

def prepare_full_pipeline(filepath):
    """Complete preprocessing pipeline"""
    df = load_data(filepath)
    df = add_type_trip(df)
    df = preprocess_features(df)
    df_encoded = encode_features(df)
    feature_cols = get_feature_columns(df_encoded)
    
    X = df_encoded[feature_cols].values
    X_scaled, scaler = scale_features(X)
    
    return df, df_encoded, feature_cols, X_scaled, scaler
