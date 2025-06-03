# Customer Churn Prediction: A Comparative Analysis of ML algorithms
Comparative analysis of ML models on a custumer churn dataset.

Models implemented and compared include:
- K-Nearest Neighbours (KNN)
- Decision Trees (DT)
- Random Forests (RF)
- Logistic Regression (LR)
- Weighted Logistic Regression (WLR)
- Extreme Learning Machine (ELM)

Resampling techniques:
- SMOTE (Synthetic Minority Oversampling Technique)
- Tomek Links

---


### **1. Dataset**
A copy of the churn dataset was downloaded from:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn


### **2.notebook**
Used for tuning and experimenting on developed models before integrating into the GUI.

### **3.models**
This folder contain my implemented ML models from scratch.

---

## **How to Run the Project**

### Step 1: Create the virtual environment

```bash
python -m venv venv
```

### Step 2: Activate the environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

This will install all the necessary libraries required for:
- Running the GUI
- Loading and evaluating models
- Visualising performance metrics
- Scripts for class imbalance

### Step 4: Launch the GUI
- Ensure you are in the root direcotry of the project 

```bash
python -m gui.app_main
```

This interactive GUI allows you to:
- Compare multiple machine learning models
- View Test set scores
- View Cross-Validation scores


### Step 6: Run Unit Tests
To run test ensure you are in the root directory of the project

Then to run individual test files:

```bash
python -m unittest tests/test_knn.py
python -m unittest tests/test_decision_tree.py
python -m unittest tests/test_model_evaluator.py
python -m unittest tests/test_lr.py
python -m unittest tests/test_weighted_lr.py
python -m unittest tests/test_elm.py
python -m unittest tests/test_random_forest.py
python -m unittest tests/test_resampling.py
python -m unittest tests/test_cross_val.py
```

Or run all tests at once:

```bash
python -m unittest discover -s tests
```





