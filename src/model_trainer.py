"""
Model Training Module for Mini Capstone ML Project
===================================================
Unified interface for training all required ML models.
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple, Optional
import config

def train_regression_models(X_train, y_train, X_test, y_test):
    """Train all regression models and return results."""
    from data_preprocessor import create_regression_pipeline
    results = {}
    
    # 1. Linear Regression
    print("\n📊 Training Linear Regression...")
    lr_model = LinearRegression()
    lr_pipe = create_regression_pipeline(lr_model)
    lr_pipe.fit(X_train, y_train)
    results['Linear Regression'] = lr_pipe
    
    # 2. Polynomial Regression
    print("📊 Training Polynomial Regression...")
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ])
    poly_pipe = create_regression_pipeline(poly_model)
    poly_pipe.fit(X_train, y_train)
    results['Polynomial Regression'] = poly_pipe
    
    # 3. Ridge Regression
    print("📊 Training Ridge Regression...")
    ridge_model = Ridge()
    ridge_pipe = create_regression_pipeline(ridge_model)
    param_grid = {'model__alpha': config.REGRESSION_PARAMS['ridge']['alpha']}
    grid_ridge = GridSearchCV(ridge_pipe, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_ridge.fit(X_train, y_train)
    results['Ridge'] = grid_ridge
    
    # 4. Lasso Regression
    print("📊 Training Lasso Regression...")
    lasso_model = Lasso(max_iter=5000)
    lasso_pipe = create_regression_pipeline(lasso_model)
    param_grid = {'model__alpha': config.REGRESSION_PARAMS['lasso']['alpha']}
    grid_lasso = GridSearchCV(lasso_pipe, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_lasso.fit(X_train, y_train)
    results['Lasso'] = grid_lasso
    
    # 5. ElasticNet
    print("📊 Training ElasticNet...")
    elastic_model = ElasticNet(max_iter=5000)
    elastic_pipe = create_regression_pipeline(elastic_model)
    param_grid = {
        'model__alpha': config.REGRESSION_PARAMS['elasticnet']['alpha'],
        'model__l1_ratio': config.REGRESSION_PARAMS['elasticnet']['l1_ratio']
    }
    grid_elastic = GridSearchCV(elastic_pipe, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_elastic.fit(X_train, y_train)
    results['ElasticNet'] = grid_elastic
    
    return results

def train_classification_models(X_train, y_train, X_test, y_test, use_smote=True):
    """Train all classification models and return results."""
    from data_preprocessor import create_classification_pipeline
    results = {}
    
    # 1. Logistic Regression
    print("\n📊 Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42)
    lr_pipe = create_classification_pipeline(lr_model, use_smote)
    param_grid = config.CLASSIFICATION_PARAMS['logistic']
    grid_lr = GridSearchCV(lr_pipe, param_grid, cv=5, scoring='f1')
    grid_lr.fit(X_train, y_train)
    results['Logistic Regression'] = grid_lr
    
    # 2. KNN
    print("📊 Training KNN...")
    knn_model = KNeighborsClassifier()
    knn_pipe = create_classification_pipeline(knn_model, use_smote)
    param_grid = config.CLASSIFICATION_PARAMS['knn']
    grid_knn = GridSearchCV(knn_pipe, param_grid, cv=5, scoring='f1')
    grid_knn.fit(X_train, y_train)
    results['KNN'] = grid_knn
    
    # 3. SVM Linear
    print("📊 Training SVM (Linear)...")
    svm_linear = SVC(kernel='linear', random_state=42, probability=True)
    svm_linear_pipe = create_classification_pipeline(svm_linear, use_smote)
    param_grid = config.CLASSIFICATION_PARAMS['svm_linear']
    grid_svm_linear = GridSearchCV(svm_linear_pipe, param_grid, cv=5, scoring='f1')
    grid_svm_linear.fit(X_train, y_train)
    results['SVM (Linear)'] = grid_svm_linear
    
    # 4. SVM RBF
    print("📊 Training SVM (RBF)...")
    svm_rbf = SVC(kernel='rbf', random_state=42, probability=True)
    svm_rbf_pipe = create_classification_pipeline(svm_rbf, use_smote)
    param_grid = config.CLASSIFICATION_PARAMS['svm_rbf']
    grid_svm_rbf = GridSearchCV(svm_rbf_pipe, param_grid, cv=5, scoring='f1')
    grid_svm_rbf.fit(X_train, y_train)
    results['SVM (RBF)'] = grid_svm_rbf
    
    # 5. Decision Tree
    print("📊 Training Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_pipe = create_classification_pipeline(dt_model, use_smote)
    param_grid = config.CLASSIFICATION_PARAMS['decision_tree']
    grid_dt = GridSearchCV(dt_pipe, param_grid, cv=5, scoring='f1')
    grid_dt.fit(X_train, y_train)
    results['Decision Tree'] = grid_dt
    
    # 6. Random Forest
    print("📊 Training Random Forest...")
    rf_model = RandomForestClassifier(random_state=42)
    rf_pipe = create_classification_pipeline(rf_model, use_smote)
    param_grid = config.CLASSIFICATION_PARAMS['random_forest']
    grid_rf = GridSearchCV(rf_pipe, param_grid, cv=5, scoring='f1')
    grid_rf.fit(X_train, y_train)
    results['Random Forest'] = grid_rf
    
    # 7. Gradient Boosting
    print("📊 Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_pipe = create_classification_pipeline(gb_model, use_smote)
    param_grid = config.CLASSIFICATION_PARAMS['gradient_boosting']
    grid_gb = GridSearchCV(gb_pipe, param_grid, cv=5, scoring='f1')
    grid_gb.fit(X_train, y_train)
    results['Gradient Boosting'] = grid_gb
    
    # 8. Stacking Classifier
    print("📊 Training Stacking Classifier...")
    base_estimators = [
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    stacking_pipe = create_classification_pipeline(stacking_model, use_smote)
    stacking_pipe.fit(X_train, y_train)
    results['Stacking'] = stacking_pipe
    
    return results

def train_clustering_models(X_scaled):
    """Train all clustering models."""
    results = {}
    
    # 1. K-Means with different k values
    for k in [2, 3, 4, 5]:
        print(f"📊 Training K-Means (k={k})...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        results[f'K-Means (k={k})'] = kmeans
    
    # 2. Agglomerative Clustering with different linkages
    for linkage in ['ward', 'complete', 'average']:
        print(f"📊 Training Agglomerative ({linkage})...")
        agg = AgglomerativeClustering(n_clusters=3, linkage=linkage)
        agg.fit(X_scaled)
        results[f'Agglomerative ({linkage})'] = agg
    
    # 3. DBSCAN
    print("📊 Training DBSCAN...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X_scaled)
    results['DBSCAN'] = dbscan
    
    # 4. PCA (Dimensionality Reduction)
    from sklearn.decomposition import PCA
    print("📊 Applying PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    results['PCA'] = {'model': pca, 'transformed': X_pca}
    
    return results
