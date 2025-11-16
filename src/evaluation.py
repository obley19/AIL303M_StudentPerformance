"""
Model Evaluation Module for Mini Capstone ML Project
====================================================
This module handles model evaluation, metrics calculation, visualization,
and performance comparison across different model types.
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
from sklearn.model_selection import learning_curve
from typing import Dict, Any, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

import config

class ModelEvaluator:
    """
    Comprehensive model evaluator for regression, classification, and clustering tasks.
    """
    
    def __init__(self, task: str = 'classification'):
        """
        Initialize the ModelEvaluator.
        
        Parameters:
        -----------
        task : str
            Type of task: 'regression', 'classification', or 'clustering'
        """
        self.task = task
        self.results = {}
        self.best_model = None
        
    def evaluate_regression_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """
        Evaluate a regression model and return metrics.
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained regression model
        X_test : array-like
            Test features
        y_test : array-like
            True target values
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict : Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'Model': model_name,
            'R2 Score': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        self.results[model_name] = metrics
        
        print(f"\n📊 {model_name} Regression Metrics:")
        print(f"   R² Score: {metrics['R2 Score']:.4f}")
        print(f"   MAE: {metrics['MAE']:.4f}")
        print(f"   RMSE: {metrics['RMSE']:.4f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        
        return metrics
    
    def evaluate_classification_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """
        Evaluate a classification model and return metrics.
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained classification model
        X_test : array-like
            Test features
        y_test : array-like
            True labels
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict : Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = None
        
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted'),
            'ROC-AUC': roc_auc if roc_auc else 'N/A'
        }
        
        self.results[model_name] = metrics
        
        print(f"\n📊 {model_name} Classification Metrics:")
        print(f"   Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   Precision: {metrics['Precision']:.4f}")
        print(f"   Recall: {metrics['Recall']:.4f}")
        print(f"   F1-Score: {metrics['F1-Score']:.4f}")
        if roc_auc:
            print(f"   ROC-AUC: {metrics['ROC-AUC']:.4f}")
        
        return metrics
    
    def evaluate_clustering_model(self, model, X, model_name: str) -> Dict:
        """
        Evaluate a clustering model and return metrics.
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained clustering model
        X : array-like
            Features used for clustering
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict : Dictionary of evaluation metrics
        """
        # Get cluster labels
        if hasattr(model, 'labels_'):
            labels = model.labels_
        else:
            labels = model.fit_predict(X)
        
        # Check if we have valid clusters (at least 2 clusters and not all noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        metrics = {'Model': model_name}
        
        if n_clusters >= 2 and len(set(labels)) >= 2:
            metrics['Silhouette Score'] = silhouette_score(X, labels)
            metrics['Davies-Bouldin Score'] = davies_bouldin_score(X, labels)
            metrics['Calinski-Harabasz Score'] = calinski_harabasz_score(X, labels)
            metrics['N_Clusters'] = n_clusters
        else:
            metrics['Silhouette Score'] = 'N/A'
            metrics['Davies-Bouldin Score'] = 'N/A'
            metrics['Calinski-Harabasz Score'] = 'N/A'
            metrics['N_Clusters'] = n_clusters
        
        self.results[model_name] = metrics
        
        print(f"\n📊 {model_name} Clustering Metrics:")
        print(f"   Number of clusters: {n_clusters}")
        if metrics['Silhouette Score'] != 'N/A':
            print(f"   Silhouette Score: {metrics['Silhouette Score']:.4f}")
            print(f"   Davies-Bouldin Score: {metrics['Davies-Bouldin Score']:.4f}")
            print(f"   Calinski-Harabasz Score: {metrics['Calinski-Harabasz Score']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, model, X_test, y_test, model_name: str):
        """
        Plot confusion matrix for classification models.
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, models: Dict, X_test, y_test):
        """
        Plot ROC curves for multiple classification models.
        """
        plt.figure(figsize=(10, 8))
        
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, model, X_test, y_test, model_name: str):
        """
        Plot residuals for regression models.
        """
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Residuals vs Fitted Values
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Fitted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residuals vs Fitted - {model_name}')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Distribution of Residuals - {model_name}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_cluster_visualization(self, X, labels, model_name: str):
        """
        Visualize clusters using PCA for dimensionality reduction.
        """
        from sklearn.decomposition import PCA
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=labels, cmap='viridis', 
                            alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter)
        plt.xlabel(f'First PC ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Second PC ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'Cluster Visualization - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves(self, model, X_train, y_train, cv=5):
        """
        Plot learning curves for a model.
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=cv,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error' if self.task == 'regression' else 'f1',
            n_jobs=-1
        )
        
        train_scores_mean = -np.mean(train_scores, axis=1) if self.task == 'regression' else np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = -np.mean(val_scores, axis=1) if self.task == 'regression' else np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label='Training score')
        plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Cross-validation score')
        
        plt.fill_between(train_sizes, 
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, 
                        alpha=0.1, color='b')
        plt.fill_between(train_sizes, 
                        val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, 
                        alpha=0.1, color='g')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, models: Dict, X_test, y_test) -> pd.DataFrame:
        """
        Compare multiple models and return comparison DataFrame.
        
        Parameters:
        -----------
        models : dict
            Dictionary of trained models {name: model}
        X_test : array-like
            Test features
        y_test : array-like
            Test labels/values
            
        Returns:
        --------
        pd.DataFrame : Comparison of all models
        """
        results_list = []
        
        for name, model in models.items():
            if self.task == 'regression':
                metrics = self.evaluate_regression_model(model, X_test, y_test, name)
            elif self.task == 'classification':
                metrics = self.evaluate_classification_model(model, X_test, y_test, name)
            else:  # clustering
                metrics = self.evaluate_clustering_model(model, X_test, name)
            
            results_list.append(metrics)
        
        comparison_df = pd.DataFrame(results_list)
        
        # Identify best model
        if self.task == 'regression':
            self.best_model = comparison_df.loc[comparison_df['R2 Score'].idxmax(), 'Model']
        elif self.task == 'classification':
            self.best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
        else:  # clustering
            valid_scores = comparison_df[comparison_df['Silhouette Score'] != 'N/A']
            if not valid_scores.empty:
                self.best_model = valid_scores.loc[valid_scores['Silhouette Score'].idxmax(), 'Model']
        
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        print(comparison_df.to_string())
        
        if self.best_model:
            print(f"\n🏆 Best Model: {self.best_model}")
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame):
        """
        Visualize model comparison results.
        """
        if self.task == 'regression':
            metrics_to_plot = ['R2 Score', 'RMSE', 'MAE']
        elif self.task == 'classification':
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        else:  # clustering
            metrics_to_plot = ['Silhouette Score', 'Davies-Bouldin Score']
        
        # Filter available metrics
        metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics_to_plot):
            # Filter out N/A values for clustering
            plot_df = comparison_df[comparison_df[metric] != 'N/A'] if metric in comparison_df.columns else comparison_df
            
            if not plot_df.empty:
                ax = axes[idx]
                bars = ax.bar(range(len(plot_df)), plot_df[metric])
                
                # Color best model differently
                if self.best_model:
                    best_idx = plot_df[plot_df['Model'] == self.best_model].index[0]
                    bars[best_idx].set_color('green')
                
                ax.set_xticks(range(len(plot_df)))
                ax.set_xticklabels(plot_df['Model'], rotation=45, ha='right')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} Comparison')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.task.capitalize()} Models Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, comparison_df: pd.DataFrame, save_path: str = None) -> str:
        """
        Generate a comprehensive evaluation report.
        """
        report = []
        report.append("="*80)
        report.append(f"MODEL EVALUATION REPORT - {self.task.upper()}")
        report.append("="*80)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append("-"*40)
        report.append(comparison_df.to_string())
        report.append("")
        
        # Best model analysis
        if self.best_model:
            report.append("BEST MODEL ANALYSIS:")
            report.append("-"*40)
            report.append(f"Best performing model: {self.best_model}")
            
            best_metrics = comparison_df[comparison_df['Model'] == self.best_model].iloc[0]
            for metric, value in best_metrics.items():
                if metric != 'Model':
                    report.append(f"  - {metric}: {value}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"✅ Report saved to {save_path}")
        
        return report_text


def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a formatted comparison table from results dictionary.
    """
    df = pd.DataFrame(results).T
    return df.round(4)
