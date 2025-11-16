"""
Model Evaluation Module for Mini Capstone ML Project
====================================================
Handles model evaluation, metrics calculation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    # Classification metrics  
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, auc,
    # Clustering metrics
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from typing import Dict, Any, Tuple, List

def evaluate_regression_models(models: Dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate all regression models."""
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        metrics = {
            'Model': name,
            'R2 Score': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        results.append(metrics)
        
        print(f"\n{name}:")
        print(f"  R²: {metrics['R2 Score']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
    
    return pd.DataFrame(results)

def evaluate_classification_models(models: Dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate all classification models."""
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted'),
            'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A'
        }
        results.append(metrics)
        
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  F1-Score: {metrics['F1-Score']:.4f}")
    
    return pd.DataFrame(results)

def evaluate_clustering_models(models: Dict, X_scaled) -> pd.DataFrame:
    """Evaluate all clustering models."""
    results = []
    
    for name, model in models.items():
        if name == 'PCA':
            continue  # Skip PCA as it's not a clustering model
            
        # Get labels
        if hasattr(model, 'labels_'):
            labels = model.labels_
        else:
            labels = model.fit_predict(X_scaled)
        
        # Check if valid clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters >= 2:
            metrics = {
                'Model': name,
                'N_Clusters': n_clusters,
                'Silhouette Score': silhouette_score(X_scaled, labels),
                'Davies-Bouldin Score': davies_bouldin_score(X_scaled, labels),
                'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, labels)
            }
        else:
            metrics = {
                'Model': name,
                'N_Clusters': n_clusters,
                'Silhouette Score': 'N/A',
                'Davies-Bouldin Score': 'N/A',
                'Calinski-Harabasz Score': 'N/A'
            }
        
        results.append(metrics)
        print(f"\n{name}:")
        print(f"  Clusters: {n_clusters}")
        if metrics['Silhouette Score'] != 'N/A':
            print(f"  Silhouette: {metrics['Silhouette Score']:.4f}")
    
    return pd.DataFrame(results)

def plot_confusion_matrices(models: Dict, X_test, y_test, figsize=(15, 10)):
    """Plot confusion matrices for classification models."""
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Prep', 'Completed'],
                   yticklabels=['No Prep', 'Completed'])
        ax.set_title(f'{name}')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Confusion Matrices - Classification Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_roc_curves(models: Dict, X_test, y_test):
    """Plot ROC curves for classification models."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_residuals(models: Dict, X_test, y_test, figsize=(15, 10)):
    """Plot residuals for regression models."""
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=figsize)
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Residuals vs Fitted
        axes[0, idx].scatter(y_pred, residuals, alpha=0.6)
        axes[0, idx].axhline(y=0, color='r', linestyle='--')
        axes[0, idx].set_xlabel('Fitted Values')
        axes[0, idx].set_ylabel('Residuals')
        axes[0, idx].set_title(f'{name}')
        axes[0, idx].grid(True, alpha=0.3)
        
        # Distribution of residuals
        axes[1, idx].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, idx].set_xlabel('Residuals')
        axes[1, idx].set_ylabel('Frequency')
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.suptitle('Residual Analysis - Regression Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_cluster_comparison(X_scaled, models: Dict):
    """Visualize clustering results using PCA."""
    from sklearn.decomposition import PCA
    
    # Reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Count valid clustering models
    cluster_models = {k: v for k, v in models.items() if k != 'PCA'}
    n_models = len(cluster_models)
    
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (name, model) in enumerate(cluster_models.items()):
        ax = axes[idx]
        
        # Get labels
        if hasattr(model, 'labels_'):
            labels = model.labels_
        else:
            labels = model.fit_predict(X_scaled)
        
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                           cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)
    
    # Hide unused
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Clustering Results Visualization (PCA)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def compare_models_barplot(df_results: pd.DataFrame, metric: str, title: str):
    """Create bar plot comparing models on a specific metric."""
    plt.figure(figsize=(12, 6))
    
    # Sort by metric
    df_sorted = df_results.sort_values(metric, ascending=False)
    
    # Create bar plot
    bars = plt.bar(range(len(df_sorted)), df_sorted[metric])
    
    # Highlight best model
    bars[0].set_color('green')
    
    plt.xticks(range(len(df_sorted)), df_sorted['Model'], rotation=45, ha='right')
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def print_classification_report_all(models: Dict, X_test, y_test):
    """Print detailed classification reports for all models."""
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Classification Report: {name}")
        print('='*60)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, 
                                   target_names=['No Prep Course', 'Completed Prep Course']))
