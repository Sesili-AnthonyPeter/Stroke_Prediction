
🧠 Brainstroke Risk Prediction

Kaggle Dataset

🚀 This repository presents a machine learning-based approach to predicting the risk of stroke. The project encompasses data preprocessing, exploratory analysis, and the implementation of multiple predictive models.

Ensure the following Python libraries are installed before you proceed:

    Numpy and Pandas for data handling and manipulation.
    Matplotlib and Seaborn for generating visualizations.
    Sklearn.preprocessing and Imblearn for scaling data and addressing class imbalance.

Getting Started

To begin working with the stroke prediction pipeline, follow these instructions:

    Clone the project repository:

    bash

git clone https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Change into the project directory:

bash

    cd stroke-prediction-dataset

    Execute the Jupyter notebook:
    Open the provided notebook to start the analysis and model-building process.

📚 Key Libraries

The following libraries were utilized throughout the project:

    Data Handling: Numpy, Pandas
    Data Visualization: Matplotlib, Seaborn
    Scaling & Resampling Techniques: Sklearn.preprocessing, Imblearn


🛠️ Data Cleaning & Preparation

Steps for preparing the data include:

    Dropping non-essential fields (id).
    Filling in missing values (BMI).
    Encoding categorical variables using dummy variables.
    Balancing the classes through random oversampling.

📊 Data Exploration

The exploratory data analysis includes:

    Studying the distribution and relationships among features.
    Creating visual representations to explore the connection between features and the target variable.

🏗️ Model Implementation

The modeling process involved:

    Splitting the dataset into training and testing sets with an 80-20 ratio.
    Training and evaluating several machine learning models: Random Forest, Decision Tree, Multilayer Perceptron, KNN, Naïve Bayes.
    Using confusion matrices, ROC curves, and AUC scores to assess model performance.

Model Accuracy:

    Random Forest: 99.38%
    Decision Tree: 97.73%
    Multilayer Perceptron: 81.49%
    KNN: 97.22%
    Naïve Bayes: 67.66%

Chosen Model: Random Forest

    The Random Forest model was selected after k-fold cross-validation (3 splits), achieving a 99.38% accuracy.

🏥 Background

Stroke occurs when the brain's blood supply is interrupted. Early prediction of stroke risk is crucial for minimizing brain damage and associated complications.
📈 Objective

    Goal: Build a model that can predict stroke risk with an accuracy exceeding 95%.
    Solution: The model leverages algorithms like K-Nearest Neighbors and Random Forest for high-accuracy predictions.


