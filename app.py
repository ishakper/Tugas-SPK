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

def load_model():
    """Load and initialize the KNN model"""
    global model_data, recommender
    
    csv_file = 'data/trip_kapal_listrik.csv'
    
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found")
        return False
    
    try:
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
    """Home page with API documentation"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>KNN Trip Recommender API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 900px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 5px; }
            h1 { color: #333; }
            .endpoint { background-color: #f9f9f9; padding: 15px; margin: 15px 0; border-left: 4px solid #007bff; }
            code { background-color: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
            .method { font-weight: bold; color: #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚢 KNN Trip Recommender API</h1>
            <p>Welcome to the KNN Trip Recommender System API. This service provides trip recommendations using K-Nearest Neighbors algorithm.</p>
            
            <h2>Available Endpoints</h2>
            
            <div class="endpoint">
                <p><span class="method">GET</span> <code>/health</code></p>
                <p>Check if the service is running</p>
                <p>Response: <code>{"status": "ok"}</code></p>
            </div>
            
            <div class="endpoint">
                <p><span class="method">POST</span> <code>/recommend/by-id</code></p>
                <p>Get recommendations based on trip ID</p>
                <p>Request body:</p>
                <pre>{
  "trip_id": "V.01",
  "n_recommendations": 5
}</pre>
            </div>
            
            <div class="endpoint">
                <p><span class="method">POST</span> <code>/recommend/by-preference</code></p>
                <p>Get recommendations based on user preferences</p>
                <p>Request body:</p>
                <pre>{
  "durasi_menit": 70,
  "total_penumpang": 10,
  "kapasitas_kursi": 15,
  "hari": "Friday",
  "jenis_hari": "Weekday",
  "shift_waktu": "Sore",
  "n_recommendations": 5
}</pre>
            </div>
        </div>
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
