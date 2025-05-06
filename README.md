# Heart Disease Classification Using Semi-Supervised Learning and Ensemble Models

## Overview
This project implements a semi-supervised learning approach to classify heart disease severity using the UCI Heart Disease dataset. The dataset contains both labeled and unlabeled samples in the `num` column, which represents disease severity from 0 (no disease) to 4 (critical). Clustering techniques are used to infer missing labels, and several supervised classifiers are trained to predict the severity of heart disease accurately.

## Objectives
- Handle missing labels using clustering
- Train multiple classification models on the completed dataset
- Compare model performance and optimize results

## Key Steps

### 1. Data Preprocessing
- Dropped irrelevant columns (`id`, `ca`, `thal`)
- Handled missing values with mean imputation
- Encoded categorical variables
- Removed outliers using the IQR method
- Standardized numerical features

### 2. Semi-Supervised Label Inference
- Applied the following clustering algorithms to infer missing `num` labels:
  - **K-Means**
  - **Spectral Clustering**
  - **Fuzzy C-Means**
- Selected K-Means based on silhouette score for label completion

### 3. Supervised Classification
- Trained several models on the completed dataset:
  - AdaBoost 
  - Gradient Boosting
  - Random Forest
  - Logistic Regression
  - Neural Network (Keras)
- Evaluated using accuracy, mean absolute error, and mean square error

### 4. Test Set Prediction
- Processed the test data using the same pipeline
- Used the best model (AdaBoost) for final prediction

## Technologies Used
- **Language**: Python
- **Libraries**: pandas, numpy, scikit-learn, seaborn, matplotlib, TensorFlow (Keras), scikit-fuzzy
- **Methods**: Semi-supervised learning, Clustering, Ensemble models, ANN, Feature Scaling, Outlier Detection

## Results
- AdaBoost achieved the highest performance across evaluation metrics
- The pipeline demonstrates an effective application of semi-supervised learning on partially labeled health data

