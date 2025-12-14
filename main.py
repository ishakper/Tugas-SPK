"""
Main script untuk KNN Trip Recommender System
Sistem rekomendasi trip privat dan open trip KM Jatim Cettar
"""

import os
import pandas as pd
from preprocessing import prepare_full_pipeline
from model import KNNRecommender, train_classifier_model
from recommendations import create_recommender

def main():
    print("=" * 70)
    print("😢 KNN TRIP RECOMMENDER SYSTEM - KM Jatim Cettar")
    print("=" * 70)
    
    # Step 1: Preprocessing
    print("\n📅 Step 1: Loading and preprocessing data...")
    csv_file = 'data/trip_kapal_listrik.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ File {csv_file} tidak ditemukan!")
        print("   Pastikan file CSV ada di folder 'data/'")
        return
    
    df, df_encoded, feature_cols, X_scaled, scaler = prepare_full_pipeline(csv_file)
    print(f"✅ Data loaded: {len(df)} trip dengan {len(feature_cols)} features")
    
    # Step 2: Train NearestNeighbors model
    print("\n🤖 Step 2: Training NearestNeighbors model (cosine similarity)...")
    nn_recommender = KNNRecommender(n_neighbors=5, metric='cosine')
    nn_recommender.fit_nearest_neighbors(X_scaled)
    
    # Step 3: Train KNN Classifier
    print("\n🎶 Step 3: Training KNN Classifier for jenis_trip...")
    knn_clf, X_test, y_test, y_pred = train_classifier_model(
        X_scaled, df_encoded['jenis_trip'].values, scaler, feature_cols
    )
    
    # Step 4: Create recommender system
    print("\n⚙️ Step 4: Setting up recommendation system...")
    recommender = create_recommender(
        df, df_encoded, X_scaled, nn_recommender.nn_model, scaler, feature_cols
    )
    print("✅ Recommendation system ready!")
    
    # Step 5: Example recommendations
    print("\n" + "=" * 70)
    print("📊 EXAMPLE RECOMMENDATIONS")
    print("=" * 70)
    
    # Contoh 1: Rekomendasi berdasarkan ID trip
    print("\n1️⃣ Berdasarkan ID Trip yang Mirip:")
    recommender.recommend_by_id('V.01', n_rekomendasi=5)
    
    # Contoh 2: Rekomendasi berdasarkan preferensi
    print("\n2️⃣ Berdasarkan Preferensi User:")
    recommender.recommend_by_preference(
        durasi_menit=70,
        total_penumpang=10,
        kapasitas_kursi=15,
        hari='Friday',
        jenis_hari='Weekday',
        shift_waktu='Sore',
        n_rekomendasi=5
    )
    
    print("\n" + "=" * 70)
    print("🙋 Project complete! Check recommendations above.")
    print("=" * 70)

if __name__ == '__main__':
    main()
