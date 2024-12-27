Brainstroke Prediction
Dataset Overview

This project leverages the Stroke Prediction Dataset from Kaggle to predict stroke risks using machine learning techniques. The dataset comprises 5110 samples with 11 features and a target column (‘stroke’).
Key Features:

    Demographic Attributes: gender, age, ever_married, Residence_type, work_type

    Medical Attributes: hypertension, heart_disease, avg_glucose_level, bmi

    Lifestyle Attributes: smoking_status

    Unique Identifier: id

Project Objective

The primary goal is to build a machine learning model capable of predicting stroke occurrences with an accuracy exceeding 95%, enabling early detection and timely intervention to minimize risks associated with brain damage.
Tools and Libraries

    Data Manipulation: Pandas, Numpy

    Data Visualization: Matplotlib, Seaborn

    Preprocessing and Modeling: Sklearn, Imblearn

Data Preprocessing

    Column Removal: Excluded irrelevant columns like id.

    Handling Missing Data: Imputed missing values in the bmi feature.

    Encoding: Applied dummy encoding for categorical variables.

    Balancing the Dataset: Used Synthetic Minority Oversampling Technique (SMOTE) to address class imbalance.

    Outlier Treatment: Identified and mitigated outliers to improve model performance.

Exploratory Data Analysis (EDA)

    Investigated distributions and correlations among features.

    Analyzed the relationship between stroke occurrences and independent variables.

    Visualized data using advanced graphical techniques for deeper insights.

Model Development
Training and Testing

    Split the data into an 80-20 ratio for training and testing.

    Evaluated five machine learning algorithms:

        Random Forest

        Decision Tree

        Multilayer Perceptron (MLP)

        K-Nearest Neighbors (KNN)

        Naïve Bayes

Evaluation Metrics

Models were assessed using:

    Accuracy

    Precision

    Recall

    F1-score

    ROC-AUC curves

Results:

    Random Forest emerged as the top-performing model:

        Accuracy: 99.38%

        Precision: 99.59%

        Recall: 100%

        F1-score: 99.79%

Hyperparameter Optimization
Grid Search

    Best Parameters: {'bootstrap': False, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}

    Results: Accuracy of 99.79%

Random Search

    Best Parameters: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': False}

    Results: Accuracy of 99.74%

Key Findings

    Random Forest achieved unparalleled accuracy and reliability, validated with k-fold cross-validation (3 splits).

    Other models, such as Decision Tree and KNN, also demonstrated high accuracy (97.73% and 97.22%, respectively).

    MLP and Naïve Bayes underperformed relative to other models.

Significance of the Study

Stroke is a critical medical condition caused by interrupted blood supply to the brain. Early detection through predictive modeling can save lives and prevent severe complications. The developed Random Forest model can serve as a vital tool for healthcare professionals in proactive stroke risk management.
Conclusion

Through rigorous preprocessing, exploratory analysis, and model optimization, the project achieved a robust predictive framework. The Random Forest model's superior performance highlights its potential in real-world healthcare applications for early stroke detection and intervention.


