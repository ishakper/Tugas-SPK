# Tugas-SPK: KNN Trip Recommender System

"Metode K-Nearest Neighbor (KNN) pada Rekomendasi Pemilihan Trip Privat dan Open Trip Kapal Listrik KM Jatim Cettar di Pantai Marina Boom Banyuwangi"

## 📋 Project Overview

Sistem rekomendasi berbasis Machine Learning menggunakan algoritma K-Nearest Neighbor (KNN) untuk merekomendasikan jenis trip (Privat vs Open Trip) yang paling sesuai dengan preferensi pengguna pada kapal listrik KM Jatim Cettar.

**Model Performance**: Akurasi 83% dalam klasifikasi jenis trip

## 🚀 Quick Start

### 1. Clone Repository ke VS Code

```bash
git clone https://github.com/ishakper/Tugas-SPK.git
cd Tugas-SPK
```

### 2. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data

- Pastikan file `trip_kapal_listrik.csv` sudah ada di folder `data/`
- Struktur folder:
  ```
  Tugas-SPK/
  ├── data/
  │   └── trip_kapal_listrik.csv
  ├── main.py
  ├── preprocessing.py
  ├── model.py
  ├── recommendations.py
  ├── requirements.txt
  └── README.md
  ```

### 5. Run Program

```bash
python main.py
```

## 📁 Project Structure

- **main.py** - Entry point untuk menjalankan sistem rekomendasi
- **preprocessing.py** - Module untuk data loading, feature engineering, dan encoding
- **model.py** - KNN classifier dan training functions
- **recommendations.py** - TripRecommender class untuk memberikan rekomendasi
- **requirements.txt** - Python dependencies

## 🔧 Module Descriptions

### preprocessing.py
- `load_data()` - Baca CSV file
- `add_type_trip()` - Tambah kolom trip type (80% Open, 20% Privat)
- `preprocess_features()` - Feature engineering (durasi, persentase isi, shift waktu, dll)
- `encode_features()` - One-hot encoding untuk fitur kategori
- `prepare_full_pipeline()` - Complete preprocessing pipeline

### model.py
- `KNNRecommender` - Class untuk KNN model
- `train_classifier_model()` - Training KNN classifier dengan train-test split

### recommendations.py
- `TripRecommender` - Class untuk rekomendasi trip
- `recommend_by_id()` - Rekomendasi berdasarkan trip ID yang mirip
- `recommend_by_preference()` - Rekomendasi berdasarkan preferensi user

## 📊 Features Digunakan

1. **Numerical Features**:
   - durasi_menit
   - total_penumpang
   - kapasitas_kursi
   - persentase_isi

2. **Categorical Features** (One-hot encoded):
   - hari (day of week)
   - jenis_hari (Weekday/Weekend)
   - shift_waktu (Pagi/Siang/Sore)

## 🎯 Model Performance

- **Algorithm**: K-Nearest Neighbors (KNN) dengan Cosine Similarity
- **K Value**: 5 neighbors
- **Metric**: Cosine Distance
- **Train-Test Split**: 80-20
- **Accuracy**: 83%

## 💡 Usage Examples

### Rekomendasi berdasarkan Trip ID

```python
recommender.recommend_by_id('V.01', n_rekomendasi=5)
```

### Rekomendasi berdasarkan Preferensi

```python
recommender.recommend_by_preference(
    durasi_menit=70,
    total_penumpang=10,
    kapasitas_kursi=15,
    hari='Friday',
    jenis_hari='Weekday',
    shift_waktu='Sore',
    n_rekomendasi=5
)
```

## 📝 Author

**ishakper** - KNN Trip Recommendation System

## 📄 License

This project is open source and available under the MIT License.
