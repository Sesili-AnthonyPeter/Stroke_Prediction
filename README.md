# 🧠 Brainstroke Prediction

[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

## Overview
Predicting stroke risk using machine learning with various preprocessing and modeling techniques.

## 📚 Libraries Used
- **Dataset Processing**: `Numpy`, `Pandas`
- **Graphical Representation**: `Matplotlib`, `Seaborn`
- **Scaling and Oversampling**: `Sklearn.preprocessing`, `Imblearn`

## 🛠️ Preprocessing
- Removed unnecessary columns (`id`).
- Checked and imputed missing values (`BMI`).
- Converted categorical data for dummy encoding.
- Balanced dataset using random oversampling.

## 📊 EDA (Exploratory Data Analysis)
- Analyzed attribute distributions and correlations.
- Plotted relation of target attribute to other attributes.

## 🏗️ Model Building
- Split data into training and testing sets (80-20 split).
- Applied various ML models: `Random Forest`, `Decision Tree`, `Multilayer Perceptron`, `KNN`, `Naïve Bayes`.
- Evaluated models using confusion matrix, ROC, and AUC scores.

### Model Accuracies:
- **Random Forest**: 99.38%
- **Decision Tree**: 97.73%
- **Multilayer Perceptron**: 81.49%
- **KNN**: 97.22%
- **Naïve Bayes**: 67.66%

### Chosen Model: **Random Forest**
- Validated using k-fold (3 splits), achieving an accuracy of 99.38%.

## 🏥 Background
Stroke is caused by disrupted blood supply to the brain. Early detection can save lives by preventing severe brain damage and complications.

## 📈 Business Understanding
**Objective**: Predict the likelihood of a stroke using machine learning with over 95% accuracy.
**Solution**: Developed a model using K-Nearest Neighbors and other algorithms.

## 📂 Data Understanding
**Dataset**: [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle.
- **5110 samples**
- **11 features**
- **1 target column (stroke)**

### Features:
- `id`: Unique identifier
- `gender`: Patient's gender
- `age`: Age of the patient
- `hypertension`: Hypertension status
- `heart_disease`: Heart disease status
- `ever_married`: Marital status
- `work_type`: Employment type
- `Residence_type`: Living area
- `avg_glucose_level`: Average glucose level
- `bmi`: Body Mass Index
- `smoking_status`: Smoking status

## 🔧 Data Preparation
- Imputed missing `bmi` values.
- Removed unnecessary columns (`id`).
- Handled outliers and standardized features.
- Applied SMOTE for balanced resampling.

## 🤖 Modeling
Used hyperparameter tuning for `Random Forest`, `Decision Tree`, `Multilayer Perceptron`, `KNN`, and `Naïve Bayes`. Achieved the highest performance with Random Forest:

### Grid Search Results:
- **Best Parameters**: `{'bootstrap': False, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}`
- **Accuracy**: 99.79%
- **Precision**: 99.59%
- **Recall**: 100%
- **F1-score**: 99.79%

### Random Search Results:
- **Best Parameters**: `{'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': False}`
- **Accuracy**: 99.74%
- **Precision**: 99.49%
- **Recall**: 100%
- **F1-score**: 99.74%

## 🏁 Conclusion

The Random Forest model achieved the best performance for stroke prediction, with accuracy, precision, recall, and F1-score all exceeding 99%. Effective preprocessing and thorough evaluation validated the model's reliability. This high-performance model can significantly aid in early stroke detection, enabling timely interventions and improving healthcare outcomes.


