"""
Utility Functions for Mini Capstone ML Project
==============================================
General utility functions for visualization, data analysis,
and helper operations used across the project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import config

# Set default plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(config.COLOR_PALETTE)

def set_plot_style():
    """Set consistent plotting style across the project."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style(config.SEABORN_STYLE)
    sns.set_palette(config.COLOR_PALETTE)
    plt.rcParams['figure.figsize'] = config.FIGURE_SIZE
    plt.rcParams['figure.dpi'] = config.DPI

def save_model(model, model_name: str, directory: str = config.MODELS_DIR):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model to save
    model_name : str
        Name for the saved model file
    directory : str
        Directory to save the model
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{model_name}.joblib")
    joblib.dump(model, filepath)
    print(f"✅ Model saved: {filepath}")

def load_model(model_name: str, directory: str = config.MODELS_DIR):
    """
    Load a saved model from disk.
    
    Parameters:
    -----------
    model_name : str
        Name of the saved model file
    directory : str
        Directory containing the model
        
    Returns:
    --------
    sklearn estimator : Loaded model
    """
    filepath = os.path.join(directory, f"{model_name}.joblib")
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        print(f"✅ Model loaded: {filepath}")
        return model
    else:
        print(f"❌ Model not found: {filepath}")
        return None

def save_results(results: Dict, filename: str = None, directory: str = config.RESULTS_DIR):
    """
    Save evaluation results to JSON file.
    
    Parameters:
    -----------
    results : dict
        Results dictionary to save
    filename : str
        Name for the results file
    directory : str
        Directory to save results
    """
    os.makedirs(directory, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
    
    filepath = os.path.join(directory, filename)
    
    # Convert any numpy types to Python types
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            results_serializable[key] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in value.items()
            }
        else:
            results_serializable[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    
    print(f"✅ Results saved: {filepath}")

def plot_correlation_matrix(df: pd.DataFrame, features: List[str] = None, 
                           figsize: Tuple = (12, 10), save_path: str = None):
    """
    Plot correlation matrix heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    features : list, optional
        List of features to include. If None, uses all numeric columns
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[features].corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, 
                square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved: {save_path}")
    
    plt.show()

def plot_feature_importance(model, feature_names: List[str], top_n: int = 20,
                          figsize: Tuple = (10, 8), save_path: str = None):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained tree-based model with feature_importances_
    feature_names : list
        Names of the features
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if not hasattr(model, 'feature_importances_'):
        print("❌ Model doesn't have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=figsize)
    plt.title('Feature Importance', fontsize=16, fontweight='bold')
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved: {save_path}")
    
    plt.show()

def plot_distribution(df: pd.DataFrame, features: List[str], 
                     figsize: Tuple = (15, 10), save_path: str = None):
    """
    Plot distribution of features using histograms and KDE.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    features : list
        List of features to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        df[feature].hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add KDE
        ax2 = ax.twinx()
        df[feature].plot(kind='kde', ax=ax2, color='red', alpha=0.5)
        ax2.set_ylabel('Density')
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved: {save_path}")
    
    plt.show()

def plot_pairwise_relationships(df: pd.DataFrame, features: List[str], 
                               target: str = None, figsize: Tuple = (15, 15),
                               save_path: str = None):
    """
    Create pairplot to show pairwise relationships between features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    features : list
        List of features to include
    target : str, optional
        Target variable for coloring
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    plot_df = df[features + [target]] if target else df[features]
    
    plt.figure(figsize=figsize)
    if target:
        sns.pairplot(plot_df, hue=target, diag_kind='kde', 
                    plot_kws={'alpha': 0.6}, height=3)
    else:
        sns.pairplot(plot_df, diag_kind='kde', 
                    plot_kws={'alpha': 0.6}, height=3)
    
    plt.suptitle('Pairwise Feature Relationships', y=1.02, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved: {save_path}")
    
    plt.show()

def print_model_summary(model, X_train, y_train, X_test, y_test, task: str = 'classification'):
    """
    Print a comprehensive summary of model performance.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_train, y_train : arrays
        Training data
    X_test, y_test : arrays
        Test data
    task : str
        Type of task
    """
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    
    # Model information
    print(f"\nModel Type: {model.__class__.__name__}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Performance metrics
    if task == 'classification':
        from sklearn.metrics import accuracy_score, f1_score
        train_score = accuracy_score(y_train, model.predict(X_train))
        test_score = accuracy_score(y_test, model.predict(X_test))
        f1 = f1_score(y_test, model.predict(X_test), average='weighted')
        
        print(f"\nTraining Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        # Check for overfitting
        if train_score - test_score > 0.1:
            print("⚠️ Warning: Possible overfitting detected!")
            
    elif task == 'regression':
        from sklearn.metrics import r2_score, mean_squared_error
        train_score = r2_score(y_train, model.predict(X_train))
        test_score = r2_score(y_test, model.predict(X_test))
        test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        
        print(f"\nTraining R² Score: {train_score:.4f}")
        print(f"Test R² Score: {test_score:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        
        # Check for overfitting
        if train_score - test_score > 0.1:
            print("⚠️ Warning: Possible overfitting detected!")
    
    print("="*80)

def create_experiment_tracker():
    """
    Create an experiment tracking DataFrame.
    
    Returns:
    --------
    pd.DataFrame : Empty DataFrame for tracking experiments
    """
    columns = [
        'Experiment_ID', 'Timestamp', 'Model', 'Task',
        'Parameters', 'Train_Score', 'Val_Score', 'Test_Score',
        'Training_Time', 'Notes'
    ]
    return pd.DataFrame(columns=columns)

def log_experiment(tracker: pd.DataFrame, model_name: str, task: str,
                  parameters: Dict, scores: Dict, training_time: float,
                  notes: str = "") -> pd.DataFrame:
    """
    Log an experiment to the tracker.
    
    Parameters:
    -----------
    tracker : pd.DataFrame
        Experiment tracking DataFrame
    model_name : str
        Name of the model
    task : str
        Type of task
    parameters : dict
        Model parameters
    scores : dict
        Performance scores
    training_time : float
        Training time in seconds
    notes : str
        Additional notes
        
    Returns:
    --------
    pd.DataFrame : Updated tracker
    """
    experiment = {
        'Experiment_ID': len(tracker) + 1,
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Model': model_name,
        'Task': task,
        'Parameters': str(parameters),
        'Train_Score': scores.get('train', None),
        'Val_Score': scores.get('val', None),
        'Test_Score': scores.get('test', None),
        'Training_Time': training_time,
        'Notes': notes
    }
    
    return pd.concat([tracker, pd.DataFrame([experiment])], ignore_index=True)

def format_time(seconds: float) -> str:
    """
    Format time duration in a readable format.
    
    Parameters:
    -----------
    seconds : float
        Time in seconds
        
    Returns:
    --------
    str : Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Perform data quality checks.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check
        
    Returns:
    --------
    dict : Data quality report
    """
    report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.to_dict(),
        'numeric_summary': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {},
        'categorical_summary': {}
    }
    
    # Categorical summary
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        report['categorical_summary'][col] = {
            'unique_values': df[col].nunique(),
            'most_frequent': df[col].mode()[0] if not df[col].mode().empty else None,
            'frequency': df[col].value_counts().to_dict()
        }
    
    return report

def print_data_quality_report(report: Dict):
    """Print formatted data quality report."""
    print("\n" + "="*80)
    print("DATA QUALITY REPORT")
    print("="*80)
    
    print(f"\nDataset Shape: {report['shape'][0]} rows × {report['shape'][1]} columns")
    print(f"Duplicate Rows: {report['duplicates']}")
    
    if any(report['missing_values'].values()):
        print("\nMissing Values:")
        for col, count in report['missing_values'].items():
            if count > 0:
                print(f"  - {col}: {count}")
    else:
        print("\n✅ No missing values found")
    
    print("\nData Types:")
    for col, dtype in report['dtypes'].items():
        print(f"  - {col}: {dtype}")
    
    print("="*80)
