import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Tuple
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import learning_curve

# Import custom preprocessing
from preprocessing_feature_engineering import get_preprocessed_data

def load_saved_model(filepath: str) -> Any:
    """
    Loads the trained machine learning model from a pickle file.
    
    Args:
        filepath (str): Path to the .pkl file.
        
    Returns:
        Any: The loaded Scikit-Learn model/pipeline.
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model successfully loaded from {filepath}")
        return model
    except FileNotFoundError:
        print(f"Error: Could not find model at {filepath}. Did you run model_selection.py first?")
        raise

def evaluate_and_save_predictions(y_true: pd.Series, y_pred: np.ndarray, output_csv_path: str) -> None:
    """
    Calculates accuracy and saves the predictions to a CSV file.
    
    Args:
        y_true (pd.Series): The actual target values.
        y_pred (np.ndarray): The predicted target values.
        output_csv_path (str): Where to save the predictions.
    """
    # 1. Calculate and print accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n======================================")
    print(f"FINAL TEST ACCURACY: {accuracy:.4f}")
    print(f"======================================")
    if accuracy > 0.65:
        print("✅ Audit Requirement Met: Test accuracy is > 0.65!")
    else:
        print("❌ Audit Requirement Failed: Test accuracy is <= 0.65.")

    # 2. Save predictions to CSV
    predictions_df = pd.DataFrame({'True_Cover_Type': y_true, 'Predicted_Cover_Type': y_pred})
    predictions_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved successfully to {output_csv_path}")

def display_and_save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_img_path: str) -> None:
    """
    Creates a confusion matrix, prints it as a DataFrame, and saves a heatmap image.
    
    Args:
        y_true (pd.Series): Actual labels.
        y_pred (np.ndarray): Predicted labels.
        output_img_path (str): Where to save the heatmap image.
    """
    # Get unique classes present in the data to label the axes correctly
    classes = sorted(y_true.unique())
    
    # Generate raw confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # 1. Audit Requirement: Display as DataFrame with specific index/column names
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    
    print("\n--- CONFUSION MATRIX ---")
    print(cm_df)
    
    # 2. Save as Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Forest Cover Type')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_img_path)
    plt.close()
    print(f"Confusion matrix heatmap saved to {output_img_path}")

def plot_and_save_learning_curve(model: Any, X_train: pd.DataFrame, y_train: pd.Series, output_img_path: str) -> None:
    """
    Generates and saves a learning curve plot using the training data.
    
    Args:
        model (Any): The trained model to evaluate.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        output_img_path (str): Where to save the plot.
    """
    print("\nGenerating Learning Curve (This may take a few minutes)...")
    
    # Note: Using 3 cross-validation folds and fewer training sizes to save time
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=3,
        n_jobs=-1,
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    
    plt.plot(train_sizes, test_mean, color='green', marker='s', linestyle='--', markersize=5, label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    
    plt.title('Learning Curve')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_img_path)
    plt.close()
    
    print(f"Learning curve saved to {output_img_path}")

def main() -> None:
    """
    Orchestrates the prediction, evaluation, and visualization pipeline.
    """
    try:
        # 1. Load the best model
        model = load_saved_model('../results/best_model.pkl')
        
        # 2. Load the blind test data (dynamically separates X and y)
        print("Loading test data...")
        X_test, y_test = get_preprocessed_data('../data/test.csv')
        
        # 3. Make predictions
        print("Making predictions on the test set...")
        y_pred = model.predict(X_test)
        
        # 4. Evaluate and save CSV
        evaluate_and_save_predictions(y_test, y_pred, '../results/test_predictions.csv')
        
        # 5. Generate Confusion Matrix
        
        display_and_save_confusion_matrix(y_test, y_pred, '../results/confusion_matrix_heatmap.png')
        
        # 6. Generate Learning Curve 
        # (Requires training data to show how the model learned over time)
        print("\nLoading training data to plot the learning curve...")
        X_train, y_train = get_preprocessed_data('../data/train.csv')
        
        # PRO TIP for Time Saving: 
        # If your model is a complex SVM or Random Forest, generating a learning curve on 500k rows
        # will take hours. We take a stratified sample of 10,000 rows just for the curve.
        _, X_train_sample, _, y_train_sample = \
            pd.model_selection.train_test_split(X_train, y_train, test_size=10000, stratify=y_train, random_state=42)
            
        plot_and_save_learning_curve(model, X_train_sample, y_train_sample, '../results/learning_curve_best_model.png')
        
        print("\n ALL PHASES COMPLETE! Check your 'results/' folder. ")
        
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")

if __name__ == "__main__":
    # Import train_test_split specifically for the learning curve sampling
    import sklearn.model_selection as model_selection
    pd.model_selection = model_selection
    main()