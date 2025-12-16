import numpy as np
import pandas as pd

class TripRecommender:
    """KNN-based recommendation engine for electric boat trips.
    
    Provides recommendation methods using K-Nearest Neighbors to suggest
    similar trips based on trip ID or user preferences (duration, passengers,
    day of week, time shift).
    """
    
    def __init__(self, df, df_encoded, X_scaled, nn_model, scaler, feature_cols):
        self.df = df
        self.df_encoded = df_encoded
        self.X_scaled = X_scaled
        self.nn_model = nn_model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.id_to_index = {tid: idx for idx, tid in enumerate(df['id_trip'].values)}
    
    def recommend_by_id(self, id_trip, n_rekomendasi=5):
        """Rekomendasi berdasarkan id_trip yang mirip"""
        if id_trip not in self.id_to_index:
            print(f"❌ id_trip '{id_trip}' tidak ditemukan di dataset")
            return
        
        idx = self.id_to_index[id_trip]
        trip_vector = self.X_scaled[idx].reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(
            trip_vector,
            n_neighbors=n_rekomendasi + 1
        )
        
        print(f"\n😢 Trip acuan: {id_trip}")
        print("📋 Rekomendasi trip mirip (cosine similarity):")
        
        for dist, ind in zip(distances.flatten()[1:], indices.flatten()[1:]):
            sim = 1 - dist
            trip_info = self.df.iloc[ind]
            print(f"   - {trip_info['id_trip']}, tgl: {trip_info['tanggal'].date()}, "
                  f"penumpang: {trip_info['total_penumpang']}, similarity: {sim:.3f}")
    
    def recommend_by_preference(self, durasi_menit, total_penumpang, kapasitas_kursi,
                                 hari='Friday', jenis_hari='Weekday', shift_waktu='Sore',
                                 n_rekomendasi=5):
        """Rekomendasi berdasarkan preferensi user"""
        data_dummy = self.df_encoded.iloc[0:1].copy()
        
        for col in data_dummy.columns:
            data_dummy[col] = 0
        
        data_dummy['durasi_menit'] = durasi_menit
        data_dummy['total_penumpang'] = total_penumpang
        data_dummy['kapasitas_kursi'] = kapasitas_kursi
        data_dummy['persentase_isi'] = total_penumpang / kapasitas_kursi
        
        for col in data_dummy.columns:
            if col.startswith('hari_'):
                data_dummy[col] = 1 if col == f'hari_{hari}' else 0
            if col.startswith('jenis_hari_'):
                data_dummy[col] = 1 if col == f'jenis_hari_{jenis_hari}' else 0
            if col.startswith('shift_waktu_'):
                data_dummy[col] = 1 if col == f'shift_waktu_{shift_waktu}' else 0
        
        X_user = data_dummy[self.feature_cols].values
        X_user_scaled = self.scaler.transform(X_user)
        
        distances, indices = self.nn_model.kneighbors(X_user_scaled, n_neighbors=n_rekomendasi)
        
        print(f"\n🎯 Rekomendasi untuk preferensi:")
        print(f"   Durasi: {durasi_menit} min, Penumpang: {total_penumpang}, "
              f"Hari: {hari}, Shift: {shift_waktu}")
        print("📋 Trip yang cocok:")
        
        for dist, ind in zip(distances.flatten(), indices.flatten()):
            sim = 1 - dist
            trip_info = self.df.iloc[ind]
            print(f"   - {trip_info['id_trip']}, tgl: {trip_info['tanggal'].date()}, "
                  f"penumpang: {trip_info['total_penumpang']}, similarity: {sim:.3f}")

def create_recommender(df, df_encoded, X_scaled, nn_model, scaler, feature_cols):
    """Factory function"""
    return TripRecommender(df, df_encoded, X_scaled, nn_model, scaler, feature_cols)
