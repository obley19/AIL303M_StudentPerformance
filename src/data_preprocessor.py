"""
Data Preprocessing Module for Mini Capstone ML Project
=======================================================
This module handles all data loading, cleaning, encoding, and transformation tasks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
import os
from typing import Tuple, Dict, Any, Optional, List

def load_data(filepath: str = '../data/StudentsPerformance.csv') -> pd.DataFrame:
    """Load the student performance dataset."""
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def get_preprocessor():
    """Create preprocessor for features."""
    cat_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch']
    num_cols = ['math score', 'reading score', 'writing score']
    
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ],
        remainder='drop'
    )

def prepare_classification_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Prepare data for classification task."""
    # Target: test preparation course
    y = df['test preparation course'].map({'none': 0, 'completed': 1})
    X = df.drop(columns=['test preparation course'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Classification data prepared:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")
    print(f"   Class distribution (train): {np.bincount(y_train)}")
    
    return X_train, X_test, y_train, y_test

def prepare_regression_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Prepare data for regression task (predict writing score)."""
    df = df.copy()
    
    # Binary encode test preparation course for use as feature
    df['test preparation course'] = df['test preparation course'].map({'none': 0, 'completed': 1})
    
    # Target: writing score
    y = df['writing score']
    X = df.drop(columns=['writing score'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✅ Regression data prepared:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def prepare_clustering_data(df: pd.DataFrame):
    """Prepare data for clustering task."""
    df = df.copy()
    
    # Binary encode test preparation course
    df['test preparation course'] = df['test preparation course'].map({'none': 0, 'completed': 1})
    
    # Create average score feature
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    
    # Select numerical features for clustering
    clustering_features = ['math score', 'reading score', 'writing score', 'average_score']
    X = df[clustering_features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Clustering data prepared:")
    print(f"   Samples: {X_scaled.shape[0]}")
    print(f"   Features: {X_scaled.shape[1]} ({', '.join(clustering_features)})")
    
    return X_scaled, df, clustering_features

def create_classification_pipeline(model, use_smote: bool = True):
    """Create pipeline for classification with optional SMOTE."""
    preprocessor = get_preprocessor()
    
    if use_smote:
        return ImbPipeline([
            ('preprocess', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
    else:
        return Pipeline([
            ('preprocess', preprocessor),
            ('model', model)
        ])

def create_regression_pipeline(model):
    """Create pipeline for regression."""
    preprocessor = get_preprocessor()
    
    return Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])
