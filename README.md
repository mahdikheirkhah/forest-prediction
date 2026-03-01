# Forest Cover Type Prediction Pipeline

## 📌 Project Overview

This project implements a complete, end-to-end Machine Learning pipeline to predict forest cover types based on cartographic variables. The pipeline is designed for production, strictly separating the training/validation phase from the final inference (prediction) phase to prevent data leakage and optimize compute resources.

## 📁 Repository Structure

```text
├── data/
│   ├── train.csv         # Training data (contains Cover_Type)
│   └── test.csv          # Blind test data (evaluated on the last day)
├── notebook/
│   └── EDA.ipynb         # Exploratory Data Analysis & initial feature logic
├── scripts/
│   ├── preprocessing_feature_engineering.py  # Utility for data loading & feature creation
│   ├── model_selection.py                    # Training, cross-validation, and model saving
│   └── predict.py                            # Inference, evaluation, and artifact generation
├── results/
│   ├── best_model.pkl                 # The serialized optimal model
│   ├── test_predictions.csv           # Final predictions on the test set
│   ├── confusion_matrix_heatmap.png   # Visualized confusion matrix
│   └── learning_curve_best_model.png  # Model learning curve
├── requirements.txt
└── README.md

```

## ⚙️ Setup & Installation

1. Ensure you have Python installed.
2. Create a virtual environment, activate it, and install requirements:
```bash
python3 -m venv ex00
source ex00/bin/activate
pip install -r requirements.txt

```


*Alternatively, on a Mac system:*
```bash
bash manage_env.sh

```



## 🚀 How to Run the Pipeline

To ensure maximum efficiency, the pipeline is split into distinct execution steps. Do not run the preprocessing script directly; it acts as a dynamic utility module for the other two scripts.

### Step 1: Train the Model (Model Selection)

This step loads the training data, applies feature engineering, splits the data, scales it via pipelines, and runs a 5-fold Stratified K-Fold Grid Search across 5 distinct algorithms.

```bash
python scripts/model_selection.py

```

*Note: This will output the `best_model.pkl` to the `results/` folder.*

### Step 2: Make Predictions (Inference)

This step loads the saved model and the blind test data, generates predictions, calculates final metrics, and generates the required visual artifacts.

```bash
python scripts/predict.py

```

## 📄 File Summaries & Feature Engineering

* **`EDA.ipynb`:** The foundational Jupyter Notebook used for Exploratory Data Analysis. It includes visualizations of feature distributions, checks for missing values, analysis of class imbalances across the 7 forest cover types, and correlation heatmaps. The insights gathered here directly informed the feature engineering strategy.
* **`preprocessing_feature_engineering.py`:** Handles data ingestion and applies custom feature engineering.
* *Engineered Features:* Implemented `Distance_To_Hydrology` (combining vertical and horizontal distances using the Pythagorean theorem) and a custom `Fire_Road_Dist_Diff` to capture the geographical relationship between fire points and roadways.


* **`model_selection.py`:** Orchestrates the grid search across 5 models (Logistic Regression, KNN, Random Forest, Gradient Boosting, SVM). Uses Scikit-Learn `Pipeline` objects to prevent data leakage during Cross-Validation.
* **`predict.py`:** Loads the `best_model.pkl`, runs inference on unseen data, outputs a DataFrame-formatted confusion matrix to the console, and generates performance charts.

## 📊 Requirements

This pipeline was built with strict adherence to project constraints:

* **Data Leakage Prevention:** `StandardScaler` is applied exclusively within Scikit-Learn `Pipeline` objects.
* **Test Split:** The training data is split into Train (1) and Test (1) using a `test_size=0.25` (satisfying the `< 0.33` constraint) with stratification to maintain class balance.
* **Cross-Validation:** 5-fold Stratified K-Fold CV is built directly into the `GridSearchCV` process.
* **Performance Constraints:**
* **Best Model:** `GradientBoosting` with Parameters: `learning_rate`: `0.1`, `max_depth`: `5`, `min_samples_leaf`: `20`, `n_estimators`: `100`.
* **Train Accuracy:** `0.9212` (Successfully kept below 0.98 to prove the model generalizes and does not massively overfit).
* **Test Accuracy:** `0.6731` (Successfully achieved well above the 0.65 threshold).



## ⭐ Bonus Features

* **Defensive Imputation:** A `SimpleImputer(strategy='median')` was added as the first step of all pipelines. While the training set may be clean, this ensures the production model will not crash if the final blind test set contains missing values.
* **Algorithm-Specific Scaling:** `StandardScaler` was purposefully **removed** from tree-based models (Random Forest, Gradient Boosting) as tree splitting logic is scale-invariant. This saves memory and compute time, demonstrating an understanding of the underlying mathematics of the algorithms.

## 📈 Visual Artifacts

Upon running `predict.py`, two visual artifacts are generated in the `results/` folder:

1. **Confusion Matrix Heatmap:** To identify specific class misclassifications.
2. **Learning Curve:** To visually diagnose bias vs. variance over varying training sizes.
