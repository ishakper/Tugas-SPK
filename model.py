from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

class KNNRecommender:
    """KNN model untuk rekomendasi trip"""
    
    def __init__(self, n_neighbors=5, metric='cosine'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.nn_model = None
        self.knn_clf = None
        
    def fit_nearest_neighbors(self, X_scaled):
        """Train NearestNeighbors model"""
        self.nn_model = NearestNeighbors(
            metric=self.metric,
            algorithm='brute',
            n_neighbors=self.n_neighbors
        )
        self.nn_model.fit(X_scaled)
        print(f"✅ NearestNeighbors model trained dengan {self.metric} similarity")
    
    def fit_classifier(self, X_train, y_train):
        """Train KNeighborsClassifier"""
        self.knn_clf = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=self.metric
        )
        self.knn_clf.fit(X_train, y_train)
        print(f"✅ KNN Classifier trained")
    
    def predict(self, X_test):
        """Predict using classifier"""
        return self.knn_clf.predict(X_test)
    
    def evaluate(self, y_test, y_pred):
        """Print classification report"""
        print("\n📊 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred))
        accuracy = np.mean(y_test == y_pred)
        print(f"\n🎯 Accuracy: {accuracy:.2%}")

def train_classifier_model(X_scaled, y, scaler, feature_cols, n_neighbors=5):
    """Train KNN classifier with 80-20 train-test split.
    
    Parameters:
    -----------
    X_scaled : array-like, shape (n_samples, n_features)
        Scaled feature matrix
    y : array-like, shape (n_samples,)
        Target labels (trip type: 'Open' or 'Privat')
    scaler : StandardScaler object
        Fitted scaler for feature normalization
    feature_cols : list
        List of feature column names
    n_neighbors : int, default=5
        Number of neighbors for KNN
        
    Returns:
    --------
    recommender : KNNRecommender
        Trained KNN recommender model
    X_test : array-like
        Test feature set
    y_test : array-like
        Test labels
    y_pred : array-like
        Predicted labels on test set
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    recommender = KNNRecommender(n_neighbors=n_neighbors, metric='cosine')
    recommender.fit_classifier(X_train, y_train)
    
    y_pred = recommender.predict(X_test)
    recommender.evaluate(y_test, y_pred)
    
    return recommender, X_test, y_test, y_pred
