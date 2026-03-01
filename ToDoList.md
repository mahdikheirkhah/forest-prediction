#### Phase 1: Environment & EDA (Hours 1-3)

* **Initialize the Repository:** Create the exact folder structure required (`data/`, `notebook/`, `scripts/`, `results/`).
* **Create requirements.txt:** Add `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and `jupyter`.
* **Write the EDA Notebook (`EDA.ipynb`):** Load `train.csv`. Check for missing values and look at the distribution of the target variable (forest cover types).
* **Engineer Features:** Create at least two composite features to show the auditor. Use this required formula for distance to hydrology:

$$Distance\_to\_hydrology = \sqrt{(Horizontal\_Distance\_To\_Hydrology)^2 + (Vertical\_Distance\_To\_Hydrology)^2}$$


* **Handle Scaling:** Prepare your preprocessing steps so tree-based models get unscaled data, while KNN/SVM/LogReg get scaled data.

#### Phase 2: Building the Pipeline (Hours 4-7)

* **Draft `preprocessing_feature_engineering.py`:** Move the successful feature creation and scaling logic from your Jupyter notebook into this Python script.
* **Draft `model_selection.py` (The Core):** Separate your target variable `X` and `y`.
* **Implement the Data Split:** Split `train.csv` into `Train (1)` and `Test (1)` (ensure the test ratio is < 33%).
* **Set up the Grid Search:** Configure a `GridSearchCV` using a 5-fold Stratified K-Fold cross-validation on `Train (1)`. Include parameter grids for Gradient Boosting, KNN, Random Forest, SVM, and Logistic Regression.

#### Phase 3: The Heavy Lifting (Hours 7-14)

* **Run the Grid Search:** This will take a long time. Start it running on a subset of data first (e.g., 10% of the rows) to ensure the code doesn't crash. Once it works, run the full Grid Search. **Step away, sleep, or take a break while this computes.**

#### Phase 4: Evaluation & Artifacts (Hours 14-16)

* **Check Accuracy Constraints:** Ensure your train accuracy is **< 0.98** (proving no massive overfitting). Ensure your validation accuracy is **> 0.65**.
* **Generate Confusion Matrix:** Format it as a pandas DataFrame (True labels as index, Predicted labels as columns). Save it as `results/confusion_matrix_heatmap.png`.
* **Plot Learning Curve:** Use scikit-learn's `learning_curve` function on your best model and save to `results/learning_curve_best_model.png`.
* **Save the Model:** Export your best estimator using the `pickle` library to `results/best_model.pkl`.

#### Phase 5: Final Predictions & Audit Prep (Hours 16-18)

* **Draft `predict.py`:** Load your `.pkl` model, apply your preprocessing script to the final blind `test.csv`, and generate predictions.
* **Save Predictions:** Export the final results to `results/test_predictions.csv`.
* **Write the `README.md`:** Document the project description, setup instructions, file summaries, and your final accuracy scores.
* **Rehearse the Stakeholder Pitch:** Spend 15 minutes practicing out loud why you engineered specific features and how your cross-validation setup prevents overfitting.

---

You have a solid roadmap now. Would you like me to write the initial Python code for `model_selection.py` so you have the correct cross-validation and Grid Search structure ready to go?