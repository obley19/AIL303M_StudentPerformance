# AIL303m Mini-Capstone Project: Student Performance Analysis

## 📚 Project Overview
**Course:** AIL303m Machine Learning  
**Contribution:** 30% of final grade  
**Duration:** 3 weeks  
**Team Size:** 5 members  

This project implements a comprehensive machine learning analysis on the **Students Performance in Exams** dataset, applying 15+ different ML algorithms across three distinct analytical paradigms: Regression, Classification, and Clustering.

## 🎯 Learning Outcome
**CLO10:** Implement a mini-capstone ML project that includes the steps: data collection, data wrangling, exploratory data analysis, model development, model evaluation, and reporting.

## 📊 Dataset
- **Source:** [Students Performance Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Size:** 1,000 student records
- **Features:** 8 attributes (5 categorical, 3 numerical)
- **Target Variables:**
  - Regression: `writing score`
  - Classification: `test preparation course`
  - Clustering: student profiles based on parental background and exam scores.

## 🔄 Tri-Modal Analysis Approach

### 1️⃣ **Regression Task**
**Objective:** Predict writing scores based on other features
- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression

### 2️⃣ **Classification Task**  
**Objective:** Predict test preparation course completion
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machines (Linear & RBF kernels)
- Decision Trees
- Random Forest (Bagging)
- Gradient Boosting
- XGBoost
- Stacking Classifier

**Imbalance Handling:** SMOTE implementation for balanced training

### 3️⃣ **Clustering Task**
**Objective:** Identify distinct student performance groups
- K-Means Clustering (k=2,3,4,5)
- Hierarchical Agglomerative Clustering
- DBSCAN
- Principal Component Analysis (PCA)

## 📁 Project Structure
```
project-root/
├── data/
│   └── StudentsPerformance.csv
├── notebooks/
│   ├── EDA.ipynb                  # Exploratory Data Analysis
│   ├── Regression.ipynb            # Regression models
│   ├── Classification.ipynb        # Classification models  
│   └── Unsupervised.ipynb          # Clustering analysis
├── src/
│   ├── config.py                   # Configuration settings
│   ├── data_preprocessor.py        # Data handling
│   ├── model_trainer.py            # Model training
│   ├── evaluation.py               # Evaluation metrics
│   ├── utils.py                    # Utility functions
│   └── main.py                     # Main execution
├── results/
│   ├── classification_results.csv
│   ├── regression_results.csv
│   └── clustering_results.csv
├── figures/
│   └── [visualizations]
├── models/
│   └── [saved models]
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```bash
# Run complete analysis
python src/main.py

# Or run individual notebooks
jupyter notebook notebooks/
```

## 📈 Key Results

### Data Insights
- **No missing values** in dataset
- **Strong correlation** between exam scores (r > 0.85)
- **Class imbalance:** 64% didn't complete test prep course
- **Performance gaps** based on socioeconomic indicators

### Model Performance Highlights
| Task | Best Model | Key Metric | Score |
|------|------------|------------|-------|
| Regression | [Model Name] | R² Score | 0.XX |
| Classification | [Model Name] | F1-Score | 0.XX |
| Clustering | K-Means (k=X) | Silhouette | 0.XX |

## 🛠️ Technical Implementation

### Data Preprocessing
- **Ordinal Encoding:** Parental education levels
- **One-Hot Encoding:** Gender, race/ethnicity, lunch
- **Feature Scaling:** StandardScaler for numerical features
- **Feature Engineering:** Average score, performance categories

### Model Optimization
- **GridSearchCV** for hyperparameter tuning
- **5-fold Cross-Validation** for robust evaluation
- **SMOTE** for handling class imbalance
- **Pipeline Integration** for reproducible workflows

### Evaluation Metrics
- **Regression:** R², MAE, MSE, RMSE, MAPE
- **Classification:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Clustering:** Silhouette Score, Davies-Bouldin, Calinski-Harabasz

## 📊 Visualizations
- Correlation heatmaps
- Distribution plots
- Confusion matrices
- ROC curves
- Residual plots
- Cluster visualizations (PCA)
- Learning curves
- Feature importance plots

## 📝 Deliverables
1. **Source Code:** GitHub repository with modular Python scripts
2. **Technical Report:** Comprehensive PDF with mathematical foundations
3. **Presentation:** 25-minute presentation (5 min/member)

## 👥 Team Members
1. Trinh Khai Nguyen - Team Leader and Regression Analysis Lead
2. Tran Gia Phuc - EDA and Visualization Specialist
3. Nguyen Chau Thanh Son - Classification Models Expert
4. Le Hoang Huu - Classification Models Expert
5. Phan Minh Tai - Unsupervised Learning Specialist

## 📋 Project Timeline
- **Week 1:** Data Understanding & EDA
- **Week 2:** Model Implementation & Training
- **Week 3:** Analysis, Synthesis & Reporting

## 🔍 Key Findings
1. **Multicollinearity:** High correlation among exam scores necessitates regularization
2. **Socioeconomic Factors:** Lunch type (proxy for SES) significantly impacts performance
3. **Test Prep Effectiveness:** Clear performance improvement with course completion
4. **Student Clusters:** [X] distinct performance groups identified

## 📚 References
- Scikit-learn Documentation
- Imbalanced-learn Documentation
- Course Materials: AIL303m Machine Learning
- Kaggle Dataset Documentation

## 📄 License
This project is part of academic coursework at FPT University.

## 🙏 Acknowledgments
- Course Instructor: Nguyen An Khuong
- FPT University AI Department
- Kaggle Community for dataset
