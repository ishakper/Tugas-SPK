# Deployment Guide for KNN Trip Recommender System

This document provides comprehensive instructions for deploying the KNN Trip Recommender System to various cloud platforms.

## Overview

The KNN Trip Recommender System is now ready for deployment with the following components:

- **app.py**: Flask REST API application
- **Procfile**: Heroku process definition
- **runtime.txt**: Python version specification
- **requirements.txt**: Python dependencies including Flask and gunicorn

## Deployment Options

### 1. Heroku Deployment

Heroku is the recommended platform for quick and easy deployment.

#### Prerequisites:
- Heroku CLI installed
- Git repository initialized
- Heroku account

#### Steps:

1. **Login to Heroku:**
```bash
heroku login
```

2. **Create a new Heroku app:**
```bash
heroku create your-app-name
```

3. **Deploy the application:**
```bash
git push heroku main
```

4. **Monitor logs:**
```bash
heroku logs --tail
```

5. **Access your app:**
Visit `https://your-app-name.herokuapp.com`

#### Important Notes:
- Ensure your CSV data file (`data/trip_kapal_listrik.csv`) is included in your repository
- The app will be available at `PORT` environment variable (default 5000)
- Heroku automatically reads `Procfile` and `runtime.txt`

### 2. Docker Deployment

For containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t knn-recommender .
docker run -p 5000:5000 knn-recommender
```

### 3. AWS EC2 Deployment

1. **Launch an EC2 instance** (Ubuntu 20.04)

2. **Connect and setup:**
```bash
sudo apt-get update
sudo apt-get install python3-pip python3-venv
```

3. **Clone repository:**
```bash
git clone <your-repo-url>
cd Tugas-SPK
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. **Run with gunicorn:**
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

### 4. Google Cloud Run Deployment

1. **Create a Dockerfile** (same as Docker section above)

2. **Build and push to Google Container Registry:**
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/knn-recommender
```

3. **Deploy to Cloud Run:**
```bash
gcloud run deploy knn-recommender \
  --image gcr.io/PROJECT-ID/knn-recommender \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## API Endpoints

Once deployed, access these endpoints:

### Health Check
```
GET /health
```

### Recommend by Trip ID
```
POST /recommend/by-id
Content-Type: application/json

{
  "trip_id": "V.01",
  "n_recommendations": 5
}
```

### Recommend by User Preferences
```
POST /recommend/by-preference
Content-Type: application/json

{
  "durasi_menit": 70,
  "total_penumpang": 10,
  "kapasitas_kursi": 15,
  "hari": "Friday",
  "jenis_hari": "Weekday",
  "shift_waktu": "Sore",
  "n_recommendations": 5
}
```

## Environment Variables

- **PORT**: Server port (default: 5000)
- **FLASK_ENV**: Flask environment (development/production)

## Performance Considerations

1. **Data Loading**: Model loads data on startup. For large datasets, consider caching
2. **CPU Requirements**: KNN calculations are CPU-intensive
3. **Memory**: Ensure sufficient memory for data processing
4. **Concurrency**: Use multiple gunicorn workers for concurrent requests

## Troubleshooting

### Data File Not Found
Ensure `data/trip_kapal_listrik.csv` exists in the repository root

### Import Errors
Verify all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Slow Performance
Increase gunicorn workers in Procfile:
```
web: gunicorn --workers 8 app:app
```

## Monitoring

- Use platform-specific monitoring (Heroku Metrics, CloudWatch, etc.)
- Monitor error logs and response times
- Set up alerts for application failures

## Scaling

- **Heroku**: Upgrade dyno size or add more dynos
- **AWS/GCP**: Use load balancers and auto-scaling groups
- **Docker**: Use Kubernetes for orchestration

## Security

- Never commit sensitive data (credentials, API keys)
- Use environment variables for configuration
- Enable HTTPS on production
- Implement rate limiting for API endpoints

## Support

For issues or questions, refer to the main README.md or create an issue on GitHub.
