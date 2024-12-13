# Customer Churn Prediction: A Comparative Analysis of ML algorithms

This project aims to predict customer churn using machine learning models implemented from scratch, including K-Nearest Neighbours (KNN) and Decision Trees. 
It also includes comprehensive exploratory data analysis (EDA) and an evaluation class to evaluate models performance.

### **1. Checkpoints**
These contains key checkpoint files for the main jupyter notebook that runs and compares the different models. It essentialy saves time and prevents retraining models

### **2. Dataset**
A copy of the churn dataset downloaded from https://www.kaggle.com/datasets/blastchar/telco-customer-churn.

### **3. Documents**
Contains relvant documents to project and is where the REPORT is located.

### **4. Jupyter Notebooks**
- **`churn_prediction_modeling.ipynb`**: Notebook for implementing and comparing machine learning models for churn prediction.
- **`Telecom_Churn_EDA.ipynb`**: Exploratory data analysis on the churn dataset, including visualisations and insights.
- **`small_dataset_test.ipynb`**: Testing machine learning models on smaller, benchmarked datasets.

## **How to Run the Project**
The outputs of all cells in the main notebooks (churn_prediction_modeling.ipynb and Telecom_Churn_EDA.ipynb) are already shown. 
Running them is not necessary unless you want to retrain models or reproduce results.
If needed, run all the cells in chronological order to avoid errors.

To run the unit tests make sure you are in the general directory of the project and use these commands in the CLI:
python -m unittest  .\tests\test_knn.py
python -m unittest  .\tests\test_decision_tree.py
python -m unittest  .\tests\test_model_evaluator.py






