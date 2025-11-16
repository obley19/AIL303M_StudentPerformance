"""
Data Preprocessing Module for Mini Capstone ML Project
=======================================================
This module handles all data loading, cleaning, encoding, and transformation tasks.
Includes functions for handling categorical variables, scaling, and dealing with
imbalanced data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
from typing import Tuple, Dict, Any, Optional, List

import config

class DataPreprocessor:
    """
    A comprehensive data preprocessor for the student performance dataset.
    Handles encoding, scaling, train-test splitting, and SMOTE for imbalanced data.
    """
    
    def __init__(self, 
                 test_size: float = config.TEST_SIZE,
                 random_state: int = config.RANDOM_STATE):
        """
        Initialize the DataPreprocessor.
        
        Parameters:
        -----------
        test_size : float, default=0.2
            Proportion of data to use for testing
        random_state : int, default=42
            Random state for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoders = {}
        self.preprocessor = None
        self.feature_names = None
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load the student performance dataset.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to the CSV file. If None, uses default from config
            
        Returns:
        --------
        pd.DataFrame : The loaded dataset
        """
        if filepath is None:
            filepath = os.path.join(config.DATA_DIR, config.DATASET_FILE)
            
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print(f"❌ Error: File not found at {filepath}")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
            
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Check for missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to check
            
        Returns:
        --------
        dict : Dictionary with column names and missing value counts
        """
        missing_counts = df.isnull().sum()
        missing_dict = missing_counts[missing_counts > 0].to_dict()
        
        if missing_dict:
            print(f"⚠️ Missing values found:")
            for col, count in missing_dict.items():
                print(f"  - {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            print("✅ No missing values found")
            
        return missing_dict
    
    def encode_ordinal_education(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode parental level of education as ordinal variable.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset
            
        Returns:
        --------
        pd.DataFrame : Dataset with encoded education column
        """
        df = df.copy()
        education_map = {level: i for i, level in enumerate(config.EDUCATION_ORDER)}
        df['parental level of education'] = df['parental level of education'].map(education_map)
        return df
    
    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create a column transformer for preprocessing.
        
        Returns:
        --------
        ColumnTransformer : Preprocessor for the features
        """
        # Identify categorical columns (excluding the already encoded education)
        cat_cols = [col for col in config.CATEGORICAL_FEATURES 
                   if col != 'parental level of education']
        
        # Add education as numerical since it's already ordinal encoded
        num_cols = config.NUMERICAL_FEATURES + ['parental level of education']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
                ('num', StandardScaler(), num_cols)
            ],
            remainder='drop'
        )
        
        return self.preprocessor
    
    def prepare_regression_data(self, df: pd.DataFrame) -> Tuple:
        """
        Prepare data for regression task (predict writing score).
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset
            
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        df = df.copy()
        
        # Encode education
        df = self.encode_ordinal_education(df)
        
        # Binary encode test preparation course for use as feature
        df['test preparation course'] = df['test preparation course'].map({
            'none': 0, 
            'completed': 1
        })
        
        # Prepare features and target
        target = config.REGRESSION_TARGET
        features = [col for col in df.columns if col != target]
        
        X = df[features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"✅ Regression data prepared:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Testing: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_classification_data(self, 
                                   df: pd.DataFrame, 
                                   use_smote: bool = True) -> Tuple:
        """
        Prepare data for classification task (predict test preparation course).
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset
        use_smote : bool, default=True
            Whether to apply SMOTE for balancing
            
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        df = df.copy()
        
        # Encode education
        df = self.encode_ordinal_education(df)
        
        # Prepare target
        y = df[config.CLASSIFICATION_TARGET].map({'none': 0, 'completed': 1})
        
        # Prepare features (exclude target column)
        X = df.drop(columns=[config.CLASSIFICATION_TARGET])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y
        )
        
        print(f"✅ Classification data prepared:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Testing: {X_test.shape[0]} samples")
        print(f"   Class distribution (train): {np.bincount(y_train)}")
        
        if use_smote:
            # Apply SMOTE only to training data
            print("   Applying SMOTE to balance classes...")
            
            # First transform the data
            preprocessor = self.create_preprocessor()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            # Apply SMOTE
            smote = SMOTE(random_state=self.random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(
                X_train_transformed, y_train
            )
            
            print(f"   After SMOTE: {np.bincount(y_train_balanced)}")
            
            return X_train_balanced, X_test_transformed, y_train_balanced, y_test
        
        return X_train, X_test, y_train, y_test
    
    def prepare_clustering_data(self, df: pd.DataFrame) -> Tuple:
        """
        Prepare data for clustering task.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset
            
        Returns:
        --------
        tuple : (X_scaled, original_data, feature_names)
        """
        df = df.copy()
        
        # Encode education
        df = self.encode_ordinal_education(df)
        
        # Binary encode test preparation course
        df['test preparation course'] = df['test preparation course'].map({
            'none': 0, 
            'completed': 1
        })
        
        # Create average score feature for clustering
        df['average_score'] = df[config.NUMERICAL_FEATURES].mean(axis=1)
        
        # Select features for clustering
        clustering_features = config.NUMERICAL_FEATURES + ['average_score']
        X = df[clustering_features]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✅ Clustering data prepared:")
        print(f"   Samples: {X_scaled.shape[0]}")
        print(f"   Features: {X_scaled.shape[1]} ({', '.join(clustering_features)})")
        
        return X_scaled, df, clustering_features
    
    def create_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset
            
        Returns:
        --------
        pd.DataFrame : Dataset with engineered features
        """
        df = df.copy()
        
        # Average score across all subjects
        df['average_score'] = df[config.NUMERICAL_FEATURES].mean(axis=1)
        
        # Standard deviation of scores (consistency measure)
        df['score_std'] = df[config.NUMERICAL_FEATURES].std(axis=1)
        
        # Total score
        df['total_score'] = df[config.NUMERICAL_FEATURES].sum(axis=1)
        
        # Performance category based on average
        df['performance_category'] = pd.cut(
            df['average_score'],
            bins=[0, 50, 70, 85, 100],
            labels=['Poor', 'Average', 'Good', 'Excellent']
        )
        
        # Binary flags for high performers
        df['high_math'] = (df['math score'] >= 80).astype(int)
        df['high_reading'] = (df['reading score'] >= 80).astype(int)
        df['high_writing'] = (df['writing score'] >= 80).astype(int)
        
        print(f"✅ Feature engineering completed: {len(df.columns) - len(config.CATEGORICAL_FEATURES) - len(config.NUMERICAL_FEATURES)} new features created")
        
        return df


def get_data_splits(task: str = 'classification', 
                    use_smote: bool = True,
                    filepath: str = None) -> Tuple:
    """
    Convenience function to get data splits for any task.
    
    Parameters:
    -----------
    task : str
        One of 'classification', 'regression', or 'clustering'
    use_smote : bool, default=True
        Whether to use SMOTE for classification
    filepath : str, optional
        Path to the dataset
        
    Returns:
    --------
    tuple : Data splits appropriate for the task
    """
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(filepath)
    
    if task == 'classification':
        return preprocessor.prepare_classification_data(df, use_smote)
    elif task == 'regression':
        return preprocessor.prepare_regression_data(df)
    elif task == 'clustering':
        return preprocessor.prepare_clustering_data(df)
    else:
        raise ValueError(f"Unknown task: {task}")
