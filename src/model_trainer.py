"""
Model Training Module for Mini Capstone ML Project
===================================================
This module provides a unified interface for training all machine learning models
required in the project, including regression, classification, and clustering models.
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            StackingClassifier)

# Clustering models
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

# Imbalanced data handling
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from typing import Dict, Any, Tuple, Optional, List
import config

class ModelTrainer:
    """
    Unified model trainer for all machine learning tasks in the project.
    """
    
    def __init__(self, 
                 task: str = 'classification',
                 random_state: int = config.RANDOM_STATE,
                 cv_folds: int = config.CV_FOLDS):
        """
        Initialize the ModelTrainer.
        
        Parameters:
        -----------
        task : str
            Type of task: 'regression', 'classification', or 'clustering'
        random_state : int
            Random state for reproducibility
        cv_folds : int
            Number of cross-validation folds
        """
        self.task = task
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.trained_models = {}
        self.training_times = {}
        self.best_params = {}
        
    def train_linear_regression(self, X_train, y_train) -> LinearRegression:
        """Train a Linear Regression model."""
        print("\n📊 Training Linear Regression...")
        start_time = time.time()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        self.training_times['Linear Regression'] = time.time() - start_time
        self.trained_models['Linear Regression'] = model
        print(f"✅ Completed in {self.training_times['Linear Regression']:.2f}s")
        
        return model
    
    def train_polynomial_regression(self, X_train, y_train, degree: int = 2) -> Pipeline:
        """Train a Polynomial Regression model."""
        print(f"\n📊 Training Polynomial Regression (degree={degree})...")
        start_time = time.time()
        
        pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', LinearRegression())
        ])
        
        pipe.fit(X_train, y_train)
        
        self.training_times['Polynomial Regression'] = time.time() - start_time
        self.trained_models['Polynomial Regression'] = pipe
        print(f"✅ Completed in {self.training_times['Polynomial Regression']:.2f}s")
        
        return pipe
    
    def train_ridge_regression(self, X_train, y_train, param_grid: Dict = None) -> Ridge:
        """Train a Ridge Regression model with hyperparameter tuning."""
        print("\n📊 Training Ridge Regression...")
        start_time = time.time()
        
        if param_grid is None:
            param_grid = {'alpha': config.REGRESSION_PARAMS['ridge']['alpha']}
        
        model = Ridge(random_state=self.random_state)
        grid_search = GridSearchCV(
            model, param_grid, cv=self.cv_folds,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params['Ridge'] = grid_search.best_params_
        self.training_times['Ridge'] = time.time() - start_time
        self.trained_models['Ridge'] = grid_search.best_estimator_
        
        print(f"✅ Best params: {self.best_params['Ridge']}")
        print(f"✅ Completed in {self.training_times['Ridge']:.2f}s")
        
        return grid_search.best_estimator_
    
    def train_lasso_regression(self, X_train, y_train, param_grid: Dict = None) -> Lasso:
        """Train a Lasso Regression model with hyperparameter tuning."""
        print("\n📊 Training Lasso Regression...")
        start_time = time.time()
        
        if param_grid is None:
            param_grid = {'alpha': config.REGRESSION_PARAMS['lasso']['alpha']}
        
        model = Lasso(random_state=self.random_state, max_iter=5000)
        grid_search = GridSearchCV(
            model, param_grid, cv=self.cv_folds,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params['Lasso'] = grid_search.best_params_
        self.training_times['Lasso'] = time.time() - start_time
        self.trained_models['Lasso'] = grid_search.best_estimator_
        
        print(f"✅ Best params: {self.best_params['Lasso']}")
        print(f"✅ Completed in {self.training_times['Lasso']:.2f}s")
        
        return grid_search.best_estimator_
    
    def train_elasticnet_regression(self, X_train, y_train, param_grid: Dict = None) -> ElasticNet:
        """Train an ElasticNet Regression model with hyperparameter tuning."""
        print("\n📊 Training ElasticNet Regression...")
        start_time = time.time()
        
        if param_grid is None:
            param_grid = config.REGRESSION_PARAMS['elasticnet']
        
        model = ElasticNet(random_state=self.random_state, max_iter=5000)
        grid_search = GridSearchCV(
            model, param_grid, cv=self.cv_folds,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params['ElasticNet'] = grid_search.best_params_
        self.training_times['ElasticNet'] = time.time() - start_time
        self.trained_models['ElasticNet'] = grid_search.best_estimator_
        
        print(f"✅ Best params: {self.best_params['ElasticNet']}")
        print(f"✅ Completed in {self.training_times['ElasticNet']:.2f}s")
        
        return grid_search.best_estimator_
    
    def train_logistic_regression(self, X_train, y_train, param_grid: Dict = None) -> LogisticRegression:
        """Train a Logistic Regression model with hyperparameter tuning."""
        print("\n📊 Training Logistic Regression...")
        start_time = time.time()
        
        if param_grid is None:
            param_grid = config.CLASSIFICATION_PARAMS['logistic']
        
        model = LogisticRegression(random_state=self.random_state)
        grid_search = GridSearchCV(
            model, param_grid, cv=self.cv_folds,
            scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params['Logistic Regression'] = grid_search.best_params_
        self.training_times['Logistic Regression'] = time.time() - start_time
        self.trained_models['Logistic Regression'] = grid_search.best_estimator_
        
        print(f"✅ Best params: {self.best_params['Logistic Regression']}")
        print(f"✅ Completed in {self.training_times['Logistic Regression']:.2f}s")
        
        return grid_search.best_estimator_
    
    def train_knn_classifier(self, X_train, y_train, param_grid: Dict = None) -> KNeighborsClassifier:
        """Train a K-Nearest Neighbors classifier."""
        print("\n📊 Training KNN Classifier...")
        start_time = time.time()
        
        if param_grid is None:
            param_grid = config.CLASSIFICATION_PARAMS['knn']
        
        model = KNeighborsClassifier()
        grid_search = GridSearchCV(
            model, param_grid, cv=self.cv_folds,
            scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params['KNN'] = grid_search.best_params_
        self.training_times['KNN'] = time.time() - start_time
        self.trained_models['KNN'] = grid_search.best_estimator_
        
        print(f"✅ Best params: {self.best_params['KNN']}")
        print(f"✅ Completed in {self.training_times['KNN']:.2f}s")
        
        return grid_search.best_estimator_
    
    def train_svm_classifier(self, X_train, y_train, kernel: str = 'rbf', param_grid: Dict = None) -> SVC:
        """Train a Support Vector Machine classifier."""
        print(f"\n📊 Training SVM Classifier (kernel={kernel})...")
        start_time = time.time()
        
        if param_grid is None:
            if kernel == 'linear':
                param_grid = config.CLASSIFICATION_PARAMS['svm_linear']
            else:
                param_grid = config.CLASSIFICATION_PARAMS['svm_rbf']
        
        model = SVC(kernel=kernel, random_state=self.random_state, probability=True)
        grid_search = GridSearchCV(
            model, param_grid, cv=self.cv_folds,
            scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params[f'SVM ({kernel})'] = grid_search.best_params_
        self.training_times[f'SVM ({kernel})'] = time.time() - start_time
        self.trained_models[f'SVM ({kernel})'] = grid_search.best_estimator_
        
        print(f"✅ Best params: {self.best_params[f'SVM ({kernel})']}")
        print(f"✅ Completed in {self.training_times[f'SVM ({kernel})']:.2f}s")
        
        return grid_search.best_estimator_
    
    def train_decision_tree(self, X_train, y_train, param_grid: Dict = None) -> DecisionTreeClassifier:
        """Train a Decision Tree classifier."""
        print("\n📊 Training Decision Tree...")
        start_time = time.time()
        
        if param_grid is None:
            param_grid = config.CLASSIFICATION_PARAMS['decision_tree']
        
        model = DecisionTreeClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(
            model, param_grid, cv=self.cv_folds,
            scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params['Decision Tree'] = grid_search.best_params_
        self.training_times['Decision Tree'] = time.time() - start_time
        self.trained_models['Decision Tree'] = grid_search.best_estimator_
        
        print(f"✅ Best params: {self.best_params['Decision Tree']}")
        print(f"✅ Completed in {self.training_times['Decision Tree']:.2f}s")
        
        return grid_search.best_estimator_
    
    def train_random_forest(self, X_train, y_train, param_grid: Dict = None) -> RandomForestClassifier:
        """Train a Random Forest classifier."""
        print("\n📊 Training Random Forest...")
        start_time = time.time()
        
        if param_grid is None:
            param_grid = config.CLASSIFICATION_PARAMS['random_forest']
        
        model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        grid_search = GridSearchCV(
            model, param_grid, cv=self.cv_folds,
            scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params['Random Forest'] = grid_search.best_params_
        self.training_times['Random Forest'] = time.time() - start_time
        self.trained_models['Random Forest'] = grid_search.best_estimator_
        
        print(f"✅ Best params: {self.best_params['Random Forest']}")
        print(f"✅ Completed in {self.training_times['Random Forest']:.2f}s")
        
        return grid_search.best_estimator_
    
    def train_gradient_boosting(self, X_train, y_train, param_grid: Dict = None) -> GradientBoostingClassifier:
        """Train a Gradient Boosting classifier."""
        print("\n📊 Training Gradient Boosting...")
        start_time = time.time()
        
        if param_grid is None:
            param_grid = config.CLASSIFICATION_PARAMS['gradient_boosting']
        
        model = GradientBoostingClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(
            model, param_grid, cv=self.cv_folds,
            scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params['Gradient Boosting'] = grid_search.best_params_
        self.training_times['Gradient Boosting'] = time.time() - start_time
        self.trained_models['Gradient Boosting'] = grid_search.best_estimator_
        
        print(f"✅ Best params: {self.best_params['Gradient Boosting']}")
        print(f"✅ Completed in {self.training_times['Gradient Boosting']:.2f}s")
        
        return grid_search.best_estimator_
    
    def train_stacking_classifier(self, X_train, y_train, base_estimators: List = None) -> StackingClassifier:
        """Train a Stacking classifier."""
        print("\n📊 Training Stacking Classifier...")
        start_time = time.time()
        
        if base_estimators is None:
            base_estimators = [
                ('lr', LogisticRegression(random_state=self.random_state, max_iter=1000)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
                ('svm', SVC(kernel='rbf', probability=True, random_state=self.random_state))
            ]
        
        model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=self.cv_folds
        )
        
        model.fit(X_train, y_train)
        
        self.training_times['Stacking'] = time.time() - start_time
        self.trained_models['Stacking'] = model
        
        print(f"✅ Completed in {self.training_times['Stacking']:.2f}s")
        
        return model
    
    def train_kmeans(self, X_train, n_clusters: int = None, param_grid: Dict = None) -> KMeans:
        """Train a K-Means clustering model."""
        print(f"\n📊 Training K-Means (k={n_clusters})...")
        start_time = time.time()
        
        if n_clusters is None:
            n_clusters = 3
        
        model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        model.fit(X_train)
        
        self.training_times[f'K-Means (k={n_clusters})'] = time.time() - start_time
        self.trained_models[f'K-Means (k={n_clusters})'] = model
        
        print(f"✅ Completed in {self.training_times[f'K-Means (k={n_clusters})']:.2f}s")
        
        return model
    
    def train_agglomerative(self, X_train, n_clusters: int = None, linkage: str = 'ward') -> AgglomerativeClustering:
        """Train an Agglomerative clustering model."""
        print(f"\n📊 Training Agglomerative Clustering (linkage={linkage})...")
        start_time = time.time()
        
        if n_clusters is None:
            n_clusters = 3
        
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        model.fit(X_train)
        
        self.training_times[f'Agglomerative ({linkage})'] = time.time() - start_time
        self.trained_models[f'Agglomerative ({linkage})'] = model
        
        print(f"✅ Completed in {self.training_times[f'Agglomerative ({linkage})']:.2f}s")
        
        return model
    
    def train_dbscan(self, X_train, eps: float = 0.5, min_samples: int = 5) -> DBSCAN:
        """Train a DBSCAN clustering model."""
        print(f"\n📊 Training DBSCAN (eps={eps}, min_samples={min_samples})...")
        start_time = time.time()
        
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(X_train)
        
        self.training_times['DBSCAN'] = time.time() - start_time
        self.trained_models['DBSCAN'] = model
        
        n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
        n_noise = list(model.labels_).count(-1)
        print(f"✅ Found {n_clusters} clusters, {n_noise} noise points")
        print(f"✅ Completed in {self.training_times['DBSCAN']:.2f}s")
        
        return model
    
    def train_all_regression_models(self, X_train, y_train) -> Dict:
        """Train all regression models."""
        print("\n" + "="*80)
        print("TRAINING ALL REGRESSION MODELS")
        print("="*80)
        
        # Train each model
        self.train_linear_regression(X_train, y_train)
        self.train_polynomial_regression(X_train, y_train)
        self.train_ridge_regression(X_train, y_train)
        self.train_lasso_regression(X_train, y_train)
        self.train_elasticnet_regression(X_train, y_train)
        
        print("\n✅ All regression models trained successfully!")
        return self.trained_models
    
    def train_all_classification_models(self, X_train, y_train) -> Dict:
        """Train all classification models."""
        print("\n" + "="*80)
        print("TRAINING ALL CLASSIFICATION MODELS")
        print("="*80)
        
        # Train each model
        self.train_logistic_regression(X_train, y_train)
        self.train_knn_classifier(X_train, y_train)
        self.train_svm_classifier(X_train, y_train, kernel='linear')
        self.train_svm_classifier(X_train, y_train, kernel='rbf')
        self.train_decision_tree(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        self.train_stacking_classifier(X_train, y_train)
        
        print("\n✅ All classification models trained successfully!")
        return self.trained_models
    
    def train_all_clustering_models(self, X_train) -> Dict:
        """Train all clustering models with different parameters."""
        print("\n" + "="*80)
        print("TRAINING ALL CLUSTERING MODELS")
        print("="*80)
        
        # Train K-Means with different k values
        for k in [2, 3, 4, 5]:
            self.train_kmeans(X_train, n_clusters=k)
        
        # Train Agglomerative with different linkage methods
        for linkage in ['ward', 'complete', 'average']:
            self.train_agglomerative(X_train, n_clusters=3, linkage=linkage)
        
        # Train DBSCAN with different parameters
        self.train_dbscan(X_train, eps=0.5, min_samples=5)
        
        print("\n✅ All clustering models trained successfully!")
        return self.trained_models
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get a summary of all trained models."""
        summary_data = {
            'Model': list(self.trained_models.keys()),
            'Training Time (s)': [self.training_times[model] for model in self.trained_models.keys()]
        }
        
        # Add best parameters if available
        if self.best_params:
            summary_data['Best Parameters'] = [
                self.best_params.get(model, 'N/A') for model in self.trained_models.keys()
            ]
        
        return pd.DataFrame(summary_data)
