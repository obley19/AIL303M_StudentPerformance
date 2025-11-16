"""
Configuration settings for the Mini Capstone ML Project
========================================================
This module contains all configuration parameters, column definitions,
hyperparameter grids, and constants used across the project.
"""

import numpy as np

# =====================================================================
# RANDOM STATE AND GENERAL SETTINGS
# =====================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# =====================================================================
# DATA COLUMNS DEFINITIONS
# =====================================================================
# Categorical features
CATEGORICAL_FEATURES = [
    'gender',
    'race/ethnicity',
    'parental level of education',
    'lunch'
]

# Numerical features (exam scores)
NUMERICAL_FEATURES = [
    'math score',
    'reading score',
    'writing score'
]

# Ordinal encoding mapping for parental education
EDUCATION_ORDER = [
    'some high school',
    'high school',
    'some college',
    "associate's degree",
    "bachelor's degree",
    "master's degree"
]

# Target columns for different tasks
REGRESSION_TARGET = 'writing score'
CLASSIFICATION_TARGET = 'test preparation course'

# =====================================================================
# HYPERPARAMETER GRIDS FOR MODEL TUNING
# =====================================================================

# Regression hyperparameter grids
REGRESSION_PARAMS = {
    'linear': {},  # No hyperparameters for basic linear regression
    
    'polynomial': {
        'degree': [2, 3, 4],
        'interaction_only': [False, True]
    },
    
    'ridge': {
        'alpha': np.logspace(-3, 3, 7)  # [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    },
    
    'lasso': {
        'alpha': np.logspace(-3, 2, 6)  # [0.001, 0.01, 0.1, 1, 10, 100]
    },
    
    'elasticnet': {
        'alpha': np.logspace(-3, 2, 6),
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
}

# Classification hyperparameter grids
CLASSIFICATION_PARAMS = {
    'logistic': {
        'model__C': np.logspace(-3, 3, 7),
        'model__solver': ['lbfgs', 'liblinear'],
        'model__max_iter': [1000]
    },
    
    'knn': {
        'model__n_neighbors': [3, 5, 7, 9, 11, 15],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan']
    },
    
    'svm_linear': {
        'model__C': np.logspace(-2, 2, 5)
    },
    
    'svm_rbf': {
        'model__C': np.logspace(-1, 3, 5),
        'model__gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 4))
    },
    
    'decision_tree': {
        'model__max_depth': [3, 5, 7, 10, None],
        'model__min_samples_split': [2, 5, 10, 20],
        'model__min_samples_leaf': [1, 2, 4, 8]
    },
    
    'random_forest': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    
    'gradient_boosting': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.3],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.8, 1.0]
    },
    
    'xgboost': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.3],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0]
    }
}

# Clustering hyperparameter grids
CLUSTERING_PARAMS = {
    'kmeans': {
        'n_clusters': range(2, 11),
        'init': ['k-means++', 'random'],
        'n_init': [10, 20],
        'max_iter': [300]
    },
    
    'agglomerative': {
        'n_clusters': range(2, 11),
        'linkage': ['ward', 'complete', 'average', 'single']
    },
    
    'dbscan': {
        'eps': np.linspace(0.1, 2.0, 20),
        'min_samples': range(2, 11)
    }
}

# =====================================================================
# EVALUATION METRICS
# =====================================================================

# Regression metrics
REGRESSION_METRICS = ['r2', 'mae', 'mse', 'rmse', 'mape']

# Classification metrics
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Clustering metrics
CLUSTERING_METRICS = ['silhouette', 'davies_bouldin', 'calinski_harabasz']

# =====================================================================
# VISUALIZATION SETTINGS
# =====================================================================
FIGURE_SIZE = (12, 8)
SEABORN_STYLE = 'whitegrid'
COLOR_PALETTE = 'husl'
DPI = 100

# Colors for different model types
MODEL_COLORS = {
    'linear': '#1f77b4',  # Blue
    'tree': '#2ca02c',    # Green
    'ensemble': '#ff7f0e', # Orange
    'svm': '#d62728',     # Red
    'knn': '#9467bd',     # Purple
    'clustering': '#8c564b' # Brown
}

# =====================================================================
# FILE PATHS
# =====================================================================
DATA_DIR = '../data'
RESULTS_DIR = '../results'
FIGURES_DIR = '../figures'
MODELS_DIR = '../models'

# Dataset filename
DATASET_FILE = 'StudentsPerformance.csv'
