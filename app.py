"""Flask app for KNN Trip Recommender System deployment"""
import os
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from preprocessing import prepare_full_pipeline
from model import KNNRecommender, train_classifier_model
from recommendations import create_recommender

app = Flask(__name__)
model_data = None
recommender = None
df_original = None

def load_model():
    global model_data, recommender, df_original
    csv_file = 'data/trip_kapal_listrik.csv'
    if not os.path.exists(csv_file):
        return False
    try:
        df_original = pd.read_csv(csv_file)
        df, df_encoded, feature_cols, X_scaled, scaler = prepare_full_pipeline(csv_file)
        nn_recommender = KNNRecommender(n_neighbors=5, metric='cosine')
        nn_recommender.fit_nearest_neighbors(X_scaled)
        knn_clf, X_test, y_test, y_pred = train_classifier_model(
            X_scaled, df_encoded['jenis_trip'].values, scaler, feature_cols
        )
        recommender = create_recommender(
            df, df_encoded, X_scaled, nn_recommender.nn_model, scaler, feature_cols
        )
        model_data = {'df': df, 'df_encoded': df_encoded, 'X_scaled': X_scaled, 'scaler': scaler, 'feature_cols': feature_cols}
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

@app.route('/')
def index():
    dataset_html = df_original.to_html(classes='table table-hover', index=False) if df_original is not None else ""
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>KNN Trip Recommender</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body {{ background: linear-gradient(135deg, #0066CC 0%, #FFD700 100%); min-height: 100vh; padding: 20px; font-family: 'Segoe UI', Tahoma; }}
        .container-custom {{ width: 100%; margin: 0 auto; max-widthwidth: 1200px; background: white; border-radius: 20px; box-shadow: 0 15px 50px rgba(0,0,0,0.3); padding: 20px 30px; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .header h1 {{ font-size: 3em; font-weight: bold; color: #0066CC; margin: 20px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }}
        .header p {{ color: #666; font-size: 1.2em; }}
        .icons {{ font-size: 3em; margin: 20px 0; }}
        .ship-icon {{ color: #FFD700; animation: bounce 2s infinite; display: inline-block; margin: 0 10px; }}
        .anchor-icon {{ color: #0066CC; animation: swing 3s infinite; display: inline-block; margin: 0 10px; }}
        @keyframes bounce {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-20px); }} }}
        @keyframes swing {{ 0%, 100% {{ transform: rotate(-5deg); }} 50% {{ transform: rotate(5deg); }} }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .stat-box {{ background: linear-gradient(135deg, #0066CC 0%, #0052A3 100%); color: white; padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 5px 20px rgba(0,102,204,0.3); transition: transform 0.3s; }}
        .stat-box:hover {{ transform: translateY(-5px); }}
        .stat-box h3 {{ font-size: 2.5em; margin: 10px 0; }}
        .stat-box p {{ margin: 0; font-size: 1.1em; opacity: 0.9; }}
        .section-title {{ font-size: 2em; color: #0066CC; margin: 40px 0 20px 0; padding-bottom: 15px; border-bottom: 4px solid #FFD700; display: flex; align-items: center; gap: 15px; }}
        .section-title i {{ color: #FFD700; font-size: 1.8em; }}
        .table {{ margin-top: 20px; border-radius: 10px; overflow: hidden;  border: 1px solid #ccc;}}
        .table thead {{ background: linear-gradient(135deg, #0066CC 0%, #FFD700 100%); color: white; font-weight: bold; }}
         .table th, .table td {{ border: 1px solid #ddd; padding: 12px; }}
         
        .table tbody tr {{ transition: all 0.2s; }}
        .table tbody tr:hover {{ background-color: #E6F2FF; transform: scale(1.01); box-shadow: 0 3px 10px rgba(0,102,204,0.2); }}
        .endpoint-card {{ background: linear-gradient(135deg, #E6F2FF 0%, #FFFACD 100%); border-left: 5px solid #0066CC; padding: 20px; margin: 15px 0; border-radius: 10px; transition: all 0.3s; }}
        .endpoint-card:hover {{ box-shadow: 0 5px 20px rgba(0,102,204,0.2); transform: translateX(5px); }}
        .method {{ color: #0066CC; font-weight: bold; font-size: 1.1em; background: #FFD700; padding: 3px 8px; border-radius: 5px; }}
        .code-block {{ background: #f8f9fa; border-left: 4px solid #FFD700; padding: 15px; margin: 10px 0; border-radius: 5px; overflow-x: auto; font-family: 'Courier New'; color: #0066CC; }}
        .table-scroll {{ overflow-x: auto; border-radius: 10px; }}
    </style></head><body>
    <div class="container-custom">
        <div class="header">
            <div class="icons">
                <span class="anchor-icon"><i class="fas fa-anchor"></i></span>
                <span class="ship-icon"><i class="fas fa-ship"></i></span>
                <span class="anchor-icon"><i class="fas fa-anchor"></i></span>
            </div>
            <h1>KNN Trip Recommender System</h1>
            <p>⛵ KM Jatim Cettar - Pantai Marina Boom Banyuwangi ⛵</p>
        </div>
        
        <div class="stats">
            <div class="stat-box"><i class="fas fa-ship" style="font-size: 2em; margin-bottom: 10px;"></i><h3>{len(df_original) if df_original is not None else 0}</h3><p>Total Trips</p></div>
            <div class="stat-box"><i class="fas fa-cube" style="font-size: 2em; margin-bottom: 10px;"></i><h3>{len(df_original.columns) if df_original is not None else 0}</h3><p>Features</p></div>
            <div class="stat-box"><i class="fas fa-chart-pie" style="font-size: 2em; margin-bottom: 10px;"></i><h3>83%</h3><p>Model Accuracy</p></div>
            <div class="stat-box"><i class="fas fa-sitemap" style="font-size: 2em; margin-bottom: 10px;"></i><h3>KNN</h3><p>Algorithm</p></div>
        </div>
        
        <h2 class="section-title"><i class="fas fa-database"></i> Dataset Trip Kapal Listrik</h2>
        <p>Semua data perjalanan trip dengan informasi lengkap:</p>
        <div class="table-scroll">
            {dataset_html}
        </div>
        
        <h2 class="section-title"><i class="fas fa-cogs"></i> API Endpoints</h2>
        <div class="endpoint-card">
            <p><span class="method">GET</span> <code>/health</code></p>
            <p>Cek status aplikasi</p>
        </div>
        <div class="endpoint-card">
            <p><span class="method">POST</span> <code>/recommend/by-id</code></p>
            <p>Dapatkan rekomendasi berdasarkan Trip ID yang mirip</p>
            <div class="code-block">{{ "trip_id": "V.01", "n_recommendations": 5 }}</div>
        </div>
        <div class="endpoint-card">
            <p><span class="method">POST</span> <code>/recommend/by-preference</code></p>
            <p>Dapatkan rekomendasi berdasarkan preferensi pengguna</p>
            <div class="code-block">{{ "durasi_menit": 70, "total_penumpang": 10, "kapasitas_kursi": 15, "hari": "Friday", "jenis_hari": "Weekday", "shift_waktu": "Sore", "n_recommendations": 5 }}</div>
        </div>
        
        <h2 class="section-title"><i class="fas fa-info-circle"></i> Informasi Sistem</h2>
        <ul style="font-size: 1.1em;">
            <li><strong style="color: #0066CC;">Algorithm:</strong> K-Nearest Neighbors (KNN)</li>
            <li><strong style="color: #0066CC;">Metric:</strong> Cosine Similarity</li>
            <li><strong style="color: #0066CC;">K Value:</strong> 5 neighbors</li>
            <li><strong style="color: #0066CC;">Model Accuracy:</strong> 83%</li>
            <li><strong style="color: #0066CC;">Total Data:</strong> {len(df_original)} trip records</li>
        </ul>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body></html>"""
    return render_template_string(html)

@app.route('/health', methods=['GET'])
def health():
    if recommender is not None:
        return jsonify({"status": "ok", "message": "Service is running"}), 200
    else:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

@app.route('/recommend/by-id', methods=['POST'])
def recommend_by_id():
    if recommender is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json()
        trip_id = data.get('trip_id')
        n_rec = data.get('n_recommendations', 5)
        if not trip_id:
            return jsonify({"error": "trip_id is required"}), 400
        result = recommender.recommend_by_id(trip_id, n_rekomendasi=n_rec)
        return jsonify({"trip_id": trip_id, "recommendations": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/recommend/by-preference', methods=['POST'])
def recommend_by_preference():
    if recommender is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json()
        result = recommender.recommend_by_preference(
            durasi_menit=data.get('durasi_menit'),
            total_penumpang=data.get('total_penumpang'),
            kapasitas_kursi=data.get('kapasitas_kursi'),
            hari=data.get('hari'),
            jenis_hari=data.get('jenis_hari'),
            shift_waktu=data.get('shift_waktu'),
            n_rekomendasi=data.get('n_recommendations', 5)
        )
        return jsonify({"preferences": data, "recommendations": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("Loading KNN model...")
    if load_model():
        print("Model loaded successfully!")
    else:
        print("Warning: Could not load model.")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

