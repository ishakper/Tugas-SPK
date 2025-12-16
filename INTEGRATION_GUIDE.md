# Integration Guide: Colab to GitHub Deployment
## KNN Trip Recommender System - Web Application

### Ringkasan

Guideline ini menjelaskan bagaimana mengintegrasikan notebook Colab kamu ke GitHub dan membuat deployed web application yang responsif untuk menampilkan semua 87 data trip.

---

## Langkah 1: Ekstrak Code dari Colab

### A. Copy Seluruh Script dari Colab

Dari Colab notebook kamu, buat beberapa script terpisah:

#### 1. `data_generator.py` - Generate Data
```python
# Kopi fungsi:
- generate_trip_dataset()
- add_derived_columns()
```

#### 2. `preprocessing.py` - Preprocessing & Pricing
```python
# Kopi:
- hitung_harga_open_trip()
- hitung_harga_privat_trip()
- hitung_multiplier_harga()
```

#### 3. `knn_model.py` - KNN Model
```python
# Kopi:
- Model training & prediction
- rekomendasi_trip_berdasarkan_id()
- rekomendasi_trip_dari_preferensi()
```

---

## Langkah 2: Setup Repository Structure

```
Tugas-SPK/
├── data/
│   └── trip_kapal_listrik.csv
├── utils/
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── knn_model.py
│   └── __init__.py
├── streamlit_app.py          ▒▒ Main deployment file
├── requirements.txt         ▒▒ Dependencies
├── INTEGRATION_GUIDE.md    ▒▒ This file
├── README.md
└── .gitignore
```

---

## Langkah 3: Deploy dengan Streamlit (Responsive)

### Install Streamlit
```bash
pip install streamlit
```

### Buat `streamlit_app.py`

**Fitur yang harus ada:**

1. **Dashboard Utama**
   - Informasi singkat tentang sistem
   - Navigation menu

2. **Data Display (Responsive)**
   - Tabel lengkap 87 data trip
   - Pagination / Show all
   - Kolom: ID, Type, Tanggal, Jam, Penumpang, Harga, Hari, Shift
   - Filter berdasarkan tipe trip, hari, shift
   - Search functionality

3. **Visualisasi Data**
   - Box plot harga per tipe
   - Histogram distribusi harga
   - Scatter: penumpang vs harga
   - Pie chart shift waktu
   - Correlation heatmap

4. **KNN Recommendations**
   - Input form untuk user preferences
   - Display top 5 rekomendasi
   - Similarity score

5. **Model Performance**
   - Confusion matrix
   - Classification metrics
   - Accuracy scores

6. **Data Export**
   - Download CSV
   - Download Excel

---

## Langkah 4: Requirements.txt

```txt
pandas>=2.0.0
numpy>=2.0.0
scikit-learn>=1.6.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
openpyxl>=3.10.0
pyarrow>=13.0.0
```

---

## Langkah 5: Jalankan Web App Locally

```bash
# Navigate ke folder project
cd Tugas-SPK

# Run streamlit
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di `http://localhost:8501`

---

## Langkah 6: Deployment ke Cloud (Streamlit Cloud / Heroku)

### Option A: Streamlit Cloud (Gratis & Mudah)

1. Push ke GitHub
2. Pergi ke https://share.streamlit.io/
3. Authorize GitHub
4. Select repository dan branch
5. Specify main file: `streamlit_app.py`
6. Deploy!

### Option B: Heroku

1. Install Heroku CLI
2. Buat `Procfile`:
```
web: streamlit run --server.port $PORT streamlit_app.py
```

3. Deploy:
```bash
heroku create nama-app-kamu
git push heroku main
```

---

## Catatan Penting ⭐

✅ **Responsif**: Interface adaptif di semua ukuran layar
✅ **Semua Data**: 87 trip ditampilkan lengkap
✅ **Filters**: Mudah mencari & memfilter data
✅ **Charts**: Visualisasi data interaktif
✅ **Export**: Download data dalam berbagai format
✅ **Performance**: Model KNN akurasi 88.89%

---

Good luck dengan deployment! 🚀
