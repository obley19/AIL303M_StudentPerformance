"""
Main Script for Mini Capstone ML Project
========================================
Execute all three tasks: Regression, Classification, and Clustering
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import config
from data_preprocessor import (
    load_data, 
    prepare_classification_data,
    prepare_regression_data,
    prepare_clustering_data
)
from model_trainer import (
    train_regression_models,
    train_classification_models,
    train_clustering_models
)
from evaluation import (
    evaluate_regression_models,
    evaluate_classification_models,
    evaluate_clustering_models,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_residuals,
    plot_cluster_comparison,
    compare_models_barplot,
    print_classification_report_all
)
from utils import (
    set_plot_style,
    save_model,
    save_results_to_csv,
    plot_correlation_heatmap,
    print_project_header,
    create_summary_table
)

def run_regression_task(df):
    """Execute regression task: Predict writing score."""
    print("\n" + "="*80)
    print("TASK 1: REGRESSION - Predicting Writing Score")
    print("="*80)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_regression_data(df)
    
    # Train models
    regression_models = train_regression_models(X_train, y_train, X_test, y_test)
    
    # Evaluate models
    results_df = evaluate_regression_models(regression_models, X_test, y_test)
    
    # Visualizations
    plot_residuals(regression_models, X_test, y_test)
    compare_models_barplot(results_df, 'R2 Score', 'Regression Models - R² Score Comparison')
    compare_models_barplot(results_df, 'RMSE', 'Regression Models - RMSE Comparison')
    
    # Save results
    save_results_to_csv(results_df, 'regression')
    
    # Save best model
    best_model_name = results_df.loc[results_df['R2 Score'].idxmax(), 'Model']
    save_model(regression_models[best_model_name], f'best_regression_{best_model_name}')
    
    print(f"\n🏆 Best Regression Model: {best_model_name}")
    print(results_df.to_string())
    
    return regression_models, results_df

def run_classification_task(df):
    """Execute classification task: Predict test preparation course completion."""
    print("\n" + "="*80)
    print("TASK 2: CLASSIFICATION - Predicting Test Preparation Course")
    print("="*80)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_classification_data(df)
    
    # Train models with SMOTE
    print("\n--- Training with SMOTE ---")
    classification_models_smote = train_classification_models(
        X_train, y_train, X_test, y_test, use_smote=True
    )
    
    # Evaluate models
    results_smote = evaluate_classification_models(
        classification_models_smote, X_test, y_test
    )
    
    # Train models without SMOTE for comparison
    print("\n--- Training without SMOTE ---")
    classification_models_no_smote = train_classification_models(
        X_train, y_train, X_test, y_test, use_smote=False
    )
    
    # Evaluate models without SMOTE
    results_no_smote = evaluate_classification_models(
        classification_models_no_smote, X_test, y_test
    )
    
    # Visualizations
    plot_confusion_matrices(classification_models_smote, X_test, y_test)
    plot_roc_curves(classification_models_smote, X_test, y_test)
    compare_models_barplot(results_smote, 'F1-Score', 'Classification Models - F1-Score Comparison (SMOTE)')
    
    # Print detailed reports
    print_classification_report_all(classification_models_smote, X_test, y_test)
    
    # Save results
    save_results_to_csv(results_smote, 'classification_smote')
    save_results_to_csv(results_no_smote, 'classification_no_smote')
    
    # Save best model
    best_model_name = results_smote.loc[results_smote['F1-Score'].idxmax(), 'Model']
    save_model(classification_models_smote[best_model_name], f'best_classification_{best_model_name}')
    
    print("\n🏆 Best Classification Model: {best_model_name}")
    print("\nWith SMOTE:")
    print(results_smote.to_string())
    print("\nWithout SMOTE:")
    print(results_no_smote.to_string())
    
    return classification_models_smote, results_smote

def run_clustering_task(df):
    """Execute clustering task: Identify student groups."""
    print("\n" + "="*80)
    print("TASK 3: CLUSTERING - Identifying Student Groups")
    print("="*80)
    
    # Prepare data
    X_scaled, df_full, feature_names = prepare_clustering_data(df)
    
    # Train models
    clustering_models = train_clustering_models(X_scaled)
    
    # Evaluate models
    results_df = evaluate_clustering_models(clustering_models, X_scaled)
    
    # Visualizations
    plot_cluster_comparison(X_scaled, clustering_models)
    
    # Compare clustering metrics
    valid_results = results_df[results_df['Silhouette Score'] != 'N/A'].copy()
    if not valid_results.empty:
        compare_models_barplot(valid_results, 'Silhouette Score', 
                              'Clustering Models - Silhouette Score Comparison')
    
    # Analyze best clustering result
    if not valid_results.empty:
        best_model_name = valid_results.loc[valid_results['Silhouette Score'].idxmax(), 'Model']
        best_model = clustering_models[best_model_name]
        
        # Get cluster labels
        if hasattr(best_model, 'labels_'):
            labels = best_model.labels_
        else:
            labels = best_model.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        df_full['Cluster'] = labels
        
        # Analyze cluster characteristics
        print(f"\n🏆 Best Clustering Model: {best_model_name}")
        print("\nCluster Statistics:")
        print(df_full.groupby('Cluster')[feature_names].mean())
        
        # Save clustered data
        df_full.to_csv('../results/clustered_data.csv', index=False)
        print("\n✅ Clustered data saved to '../results/clustered_data.csv'")
    
    # Save results
    save_results_to_csv(results_df, 'clustering')
    
    # Handle PCA separately
    if 'PCA' in clustering_models:
        pca_model = clustering_models['PCA']['model']
        X_pca = clustering_models['PCA']['transformed']
        
        print("\n📊 PCA Results:")
        print(f"Explained variance ratio: {pca_model.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(pca_model.explained_variance_ratio_):.4f}")
    
    print("\n" + results_df.to_string())
    
    return clustering_models, results_df

def main():
    """Main execution function."""
    # Set plotting style
    set_plot_style()
    
    # Print header
    print_project_header()
    
    # Load data
    df = load_data()
    
    # Display basic info
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    
    # Show correlation heatmap
    plot_correlation_heatmap(df)
    
    # Run all three tasks
    regression_models, regression_results = run_regression_task(df)
    classification_models, classification_results = run_classification_task(df)
    clustering_models, clustering_results = run_clustering_task(df)
    
    # Final summary
    print("\n" + "="*80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n📊 Summary of Results:")
    print("\n1. REGRESSION (Predicting Writing Score):")
    print(f"   Best Model: {regression_results.loc[regression_results['R2 Score'].idxmax(), 'Model']}")
    print(f"   Best R² Score: {regression_results['R2 Score'].max():.4f}")
    
    print("\n2. CLASSIFICATION (Predicting Test Prep Course):")
    print(f"   Best Model: {classification_results.loc[classification_results['F1-Score'].idxmax(), 'Model']}")
    print(f"   Best F1-Score: {classification_results['F1-Score'].max():.4f}")
    
    print("\n3. CLUSTERING (Student Groups):")
    valid_clustering = clustering_results[clustering_results['Silhouette Score'] != 'N/A']
    if not valid_clustering.empty:
        print(f"   Best Model: {valid_clustering.loc[valid_clustering['Silhouette Score'].idxmax(), 'Model']}")
        print(f"   Best Silhouette Score: {valid_clustering['Silhouette Score'].max():.4f}")
    
    print("\n✅ All models trained and evaluated successfully!")
    print("✅ Results saved to ../results/ directory")
    print("✅ Best models saved to ../models/ directory")
    
    return {
        'regression': (regression_models, regression_results),
        'classification': (classification_models, classification_results),
        'clustering': (clustering_models, clustering_results)
    }

if __name__ == "__main__":
    results = main()
