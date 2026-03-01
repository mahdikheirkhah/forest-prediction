import os
import pandas as pd
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Tuple
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import learning_curve

from preprocessing_feature_engineering import get_preprocessed_data

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_saved_model(filepath: str) -> Any:
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model successfully loaded from {filepath}")
        return model
    except FileNotFoundError:
        logging.error(f"Could not find model at {filepath}. Did you run model_selection.py first?")
        raise

def evaluate_and_save_predictions(y_true: pd.Series, y_pred: np.ndarray, output_csv_path: str) -> None:
    accuracy = accuracy_score(y_true, y_pred)
    logging.info("======================================")
    logging.info(f"FINAL TEST ACCURACY: {accuracy:.4f}")
    logging.info("======================================")
    
    if accuracy > 0.65:
        logging.info("✅ Audit Requirement Met: Test accuracy is > 0.65!")
    else:
        logging.warning("❌ Audit Requirement Failed: Test accuracy is <= 0.65.")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    predictions_df = pd.DataFrame({'True_Cover_Type': y_true, 'Predicted_Cover_Type': y_pred})
    predictions_df.to_csv(output_csv_path, index=False)
    logging.info(f"Predictions saved successfully to {output_csv_path} (Overwrites if existed)")

def display_and_save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_img_path: str) -> None:
    classes = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    
    logging.info(f"\n--- CONFUSION MATRIX ---\n{cm_df}")
    
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Forest Cover Type')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_img_path)
    plt.close()
    logging.info(f"Confusion matrix heatmap saved to {output_img_path}")

def plot_and_save_learning_curve(model: Any, X_train: pd.DataFrame, y_train: pd.Series, output_img_path: str) -> None:
    logging.info("Generating Learning Curve (This may take a few minutes)...")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model, X=X_train, y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 5), cv=3, n_jobs=-1, scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
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
    
    logging.info(f"Learning curve saved to {output_img_path}")

def main() -> None:
    logging.info("--- STARTING PREDICTION PIPELINE ---")
    try:
        model = load_saved_model('../results/best_model.pkl')
        
        logging.info("Loading blind test data...")
        X_test, y_test = get_preprocessed_data('../data/test.csv')
        
        logging.info("Making predictions on the test set...")
        y_pred = model.predict(X_test)
        
        evaluate_and_save_predictions(y_test, y_pred, '../results/test_predictions.csv')
        display_and_save_confusion_matrix(y_test, y_pred, '../results/confusion_matrix_heatmap.png')
        
        logging.info("Loading training data to plot the learning curve...")
        X_train, y_train = get_preprocessed_data('../data/train.csv')
        
        _, X_train_sample, _, y_train_sample = \
            pd.model_selection.train_test_split(X_train, y_train, test_size=10000, stratify=y_train, random_state=42)
            
        plot_and_save_learning_curve(model, X_train_sample, y_train_sample, '../results/learning_curve_best_model.png')
        
        logging.info("ALL PHASES COMPLETE! Check your 'results/' folder.")
        
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    import sklearn.model_selection as model_selection
    pd.model_selection = model_selection
    main()