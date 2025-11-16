"""
Utility Functions for Mini Capstone ML Project
==============================================
General utility functions for the project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

def set_plot_style():
    """Set consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('husl')
    plt.rcParams['figure.figsize'] = (12, 8)

def save_model(model, model_name: str, directory: str = '../models'):
    """Save trained model to disk."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{model_name}.joblib")
    joblib.dump(model, filepath)
    print(f"✅ Model saved: {filepath}")

def load_model(model_name: str, directory: str = '../models'):
    """Load saved model from disk."""
    filepath = os.path.join(directory, f"{model_name}.joblib")
    if os.path.exists(filepath):
        return joblib.load(filepath)
    print(f"❌ Model not found: {filepath}")
    return None

def save_results_to_csv(results_df: pd.DataFrame, task_name: str, directory: str = '../results'):
    """Save evaluation results to CSV."""
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(directory, f"{task_name}_results_{timestamp}.csv")
    results_df.to_csv(filepath, index=False)
    print(f"✅ Results saved: {filepath}")

def plot_correlation_heatmap(df: pd.DataFrame, figsize=(10, 8)):
    """Plot correlation matrix heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(df: pd.DataFrame, features: list, figsize=(15, 5)):
    """Plot distributions of numerical features."""
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=figsize)
    
    if n_features == 1:
        axes = [axes]
    
    for idx, feature in enumerate(features):
        df[feature].hist(ax=axes[idx], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{feature} Distribution')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_summary_table(models_dict: dict, task: str) -> pd.DataFrame:
    """Create summary table of all models."""
    summary = []
    for name, model in models_dict.items():
        model_info = {
            'Model': name,
            'Type': model.__class__.__name__ if hasattr(model, '__class__') else 'Pipeline',
            'Task': task
        }
        
        # Add best params if GridSearchCV
        if hasattr(model, 'best_params_'):
            model_info['Best_Params'] = str(model.best_params_)
        
        summary.append(model_info)
    
    return pd.DataFrame(summary)

def print_project_header():
    """Print project header."""
    print("="*80)
    print("AIL303m MINI-CAPSTONE PROJECT - STUDENT PERFORMANCE ANALYSIS")
    print("="*80)
    print("Dataset: Students Performance in Exams")
    print("Tasks: Regression, Classification, Clustering")
    print("Models: 15 different ML algorithms")
    print("="*80)

def format_metrics(metrics_dict: dict) -> str:
    """Format metrics dictionary for display."""
    output = []
    for metric, value in metrics_dict.items():
        if isinstance(value, float):
            output.append(f"{metric}: {value:.4f}")
        else:
            output.append(f"{metric}: {value}")
    return " | ".join(output)
