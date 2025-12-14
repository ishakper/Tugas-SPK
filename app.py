"""Flask app for KNN Trip Recommender System deployment"""
import os
import sys
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from preprocessing import prepare_full_pipeline
from model import KNNRecommender, train_classifier_model
from recommendations import create_recommender

app = Flask(__name__)

# Global variables to store model and data
model_data = None
recommender = None
df_original = None

def load_model():
    """Load and initialize the KNN model"""
    global model_data, recommender, df_original
    
    csv_file = 'data/trip_kapal_listrik.csv'
    
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found")
        return False
    
    try:
        # Load original data
        df_original = pd.read_csv(csv_file)
        
        # Preprocessing
        df, df_encoded, feature_cols, X_scaled, scaler = prepare_full_pipeline(csv_file)
        
        # Train NearestNeighbors model
        nn_recommender = KNNRecommender(n_neighbors=5, metric='cosine')
        nn_recommender.fit_nearest_neighbors(X_scaled)
        
        # Train KNN Classifier
        knn_clf, X_test, y_test, y_pred = train_classifier_model(
            X_scaled, df_encoded['jenis_trip'].values, scaler, feature_cols
        )
        
        # Create recommender system
        recommender = create_recommender(
            df, df_encoded, X_scaled, nn_recommender.nn_model, scaler, feature_cols
        )
        
        model_data = {
            'df': df,
            'df_encoded': df_encoded,
            'X_scaled': X_scaled,
            'scaler': scaler,
            'feature_cols': feature_cols
        }
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/', methods=['GET'])
def index():
    """Home page with dataset and API documentation"""
    
    # Prepare dataset table HTML
    dataset_html = ""
    if df_original is not None:
        dataset_html = df_original.to_html(classes='table table-striped table-bordered', index=False)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>KNN Trip Recommender System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            .container-custom {{
                max-width: 1200px;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                padding: 40px;
                margin-bottom: 20px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                color: #333;
            }}
            .header h1 {{
                font-size: 2.5em;
                font-weight: bold;
                margin-bottom: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .header p {{
                color: #666;
                font-size: 1.1em;
            }}
            .section-title {{
                font-size: 1.8em;
                color: #333;
                margin: 30px 0 20px 0;
                padding-bottom: 10px;
                border-bottom: 3px solid #667eea;
            }}
            .table {{
                margin-top: 20px;
                border-collapse: collapse;
            }}
            .table thead {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .table tbody tr:hover {{
                background-color: #f5f5f5;
                transition: 0.3s;
            }}
            .table th {{
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}
            .table td {{
                padding: 10px 12px;
                border-bottom: 1px solid #ddd;
            }}
            .endpoint-card {{
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-left: 5px solid #667eea;
                padding: 20px;
                margin: 15px 0;
                border-radius: 8px;
                transition: 0.3s;
            }}
            .endpoint-card:hover {{
                transform: translateX(5px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            }}
            .method {{
                font-weight: bold;
                color: #667eea;
                font-size: 1.1em;
            }}
            .code-block {{
                background: #f4f4f4;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
            }}
            .badge-success {{
                background: #28a745;
            }}
            .table-scroll {{
                overflow-x: auto;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .stat-box h3 {{
                font-size: 2em;
                margin: 10px 0;
            }}
            .stat-box p {{
                margin: 0;
                opacity: 0.9;
            }}
        </style>
    </head>
    <body>
        <div class="container-custom">
            <div class="header">
                <h1>🚢 KNN Trip Recommender System</h1>
                <p>KM Jatim Cettar - Pantai Marina Boom Banyuwangi</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>{len(df_original) if df_original is not None else 0}</h3>
                    <p>Total Trips</p>
                </div>
                <div class="stat-box">
                    <h3>{len(df_original.columns) if df_original is not None else 0}</h3>
                    <p>Features</p>
                </div>
                <div class="stat-box">
                    <h3>83%</h3>
                    <p>Model Accuracy</p>
                </div>
                <div class="stat-box">
                    <h3>KNN</h3>
                    <p>Algorithm</p>
                </div>
            </div>
            
            <hr>
            
            <h2 class="section-title">📊 Dataset Trip Kapal Listrik</h2>
            <p>Data lengkap untuk sistem rekomendasi perjalanan trip privat dan open trip:</p>
            <div class="table-scroll">
                {dataset_html}
            </div>
            
            <hr>
            
            <h2 class="section-title">🔌 API Endpoints</h2>
            <p>Gunakan endpoint berikut untuk mendapatkan rekomendasi trip:</p>
            
            <div class="endpoint-card">
                <p><span class="method">GET</span> <code>/health</code></p>
                <p>Cek status aplikasi</p>
                <div class="code-block">
Response: {{
  "status": "ok",
  "message": "Service is running"
}}
                </div>
            </div>
            
            <div class="endpoint-card">
                <p><span class="method">POST</span> <code>/recommend/by-id</code></p>
                <p>Dapatkan rekomendasi berdasarkan Trip ID yang mirip</p>
                <div class="code-block">
{{
  "trip_id": "V.01",
  "n_recommendations": 5
}}
                </div>
            </div>
            
            <div class="endpoint-card">
                <p><span class="method">POST</span> <code>/recommend/by-preference</code></p>
                <p>Dapatkan rekomendasi berdasarkan preferensi pengguna</p>
                <div class="code-block">
{{
  "durasi_menit": 70,
  "total_penumpang": 10,
  "kapasitas_kursi": 15,
  "hari": "Friday",
  "jenis_hari": "Weekday",
  "shift_waktu": "Sore",
  "n_recommendations": 5
}}
                </div>
            </div>
            
            <hr>
            
            <h2 class="section-title">📋 Informasi Sistem</h2>
            <ul>
                <li><strong>Algorithm:</strong> K-Nearest Neighbors (KNN)</li>
                <li><strong>Metric:</strong> Cosine Similarity</li>
                <li><strong>K Value:</strong> 5 neighbors</li>
                <li><strong>Model Accuracy:</strong> 83%</li>
                <li><strong>Total Data:</strong> {len(df_original)} trip records</li>
            </ul>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if recommender is not None:
        return jsonify({"status": "ok", "message": "Service is running"}), 200
    else:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

@app.route('/recommend/by-id', methods=['POST'])
def recommend_by_id():
    """Recommend trips based on trip ID"""
    if recommender is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        trip_id = data.get('trip_id')
        n_rec = data.get('n_recommendations', 5)
        
        if not trip_id:
            return jsonify({"error": "trip_id is required"}), 400
        
        # Get recommendations
        result = recommender.recommend_by_id(trip_id, n_rekomendasi=n_rec)
        
        return jsonify({
            "trip_id": trip_id,
            "recommendations": result
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/recommend/by-preference', methods=['POST'])
def recommend_by_preference():
    """Recommend trips based on user preferences"""
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
        
        return jsonify({
            "preferences": data,
            "recommendations": result
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Load model on startup
    print("Loading KNN model...")
    if load_model():
        print("Model loaded successfully!")
    else:
        print("Warning: Could not load model. API will return errors.")
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False)
