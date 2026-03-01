import pandas as pd
import pickle
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Import custom preprocessing
from preprocessing_feature_engineering import get_preprocessed_data

def load_and_split_data(filepath: str, test_size: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads preprocessed data and splits it into training and validation sets.
    
    Args:
        filepath (str): Path to the training data CSV.
        test_size (float): Proportion of the dataset to include in the validation split (< 0.33).
        
    Returns:
        Tuple containing X_train, X_test, y_train, y_test.
    """
    try:
        print(f"Loading data from {filepath}...")
        X, y = get_preprocessed_data(filepath)
        
        # Split train (0) into Train (1) and Test (1)
        # Using stratify=y to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"Data split successful. Training size: {X_train.shape[0]}, Validation size: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error during data loading and splitting: {e}")
        raise

def get_model_pipelines() -> Dict[str, Dict[str, Any]]:
    """
    Defines the machine learning models, scaling pipelines, and hyperparameter grids.
    
    Returns:
        Dictionary containing model configurations.
    """
    # NOTE: Scaler is strictly applied ONLY to models that require it.
    # To save time during initial testing, hyperparameter grids are kept extremely small.
    # You should expand these grids when running the final overnight training.
    
    pipelines_and_grids = {
        'LogisticRegression': {
            'model': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
            'params': {'clf__C': [0.1, 1.0]}
        },
        'KNN': {
            'model': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]),
            'params': {'clf__n_neighbors': [3, 5]}
        },
        'RandomForest': {
            'model': Pipeline([('clf', RandomForestClassifier(random_state=42))]),
            'params': {'clf__n_estimators': [50, 100], 'clf__max_depth': [10, None]}
        },
        'GradientBoosting': {
            'model': Pipeline([('clf', GradientBoostingClassifier(random_state=42))]),
            'params': {'clf__n_estimators': [50], 'clf__learning_rate': [0.1]}
        },
        'SVM': {
            'model': Pipeline([('scaler', StandardScaler()), ('clf', SVC(random_state=42))]),
            'params': {'clf__C': [1.0], 'clf__kernel': ['rbf']}
        }
    }
    return pipelines_and_grids

def perform_grid_search(X_train: pd.DataFrame, y_train: pd.Series, models_config: Dict[str, Dict[str, Any]]) -> Tuple[Any, str]:
    """
    Iterates through model configurations, performs Grid Search with Cross Validation, 
    and returns the best performing model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        models_config (Dict): The pipelines and parameter grids to evaluate.
        
    Returns:
        Tuple containing the (best_model_object, best_model_name).
    """
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_overall_model = None
    best_overall_score = 0.0
    best_overall_name = ""

    for model_name, config in models_config.items():
        print(f"\n--- Starting Grid Search for {model_name} ---")
        try:
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=cv_strategy,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"[{model_name}] Best Parameters: {grid_search.best_params_}")
            print(f"[{model_name}] Best CV Accuracy: {grid_search.best_score_:.4f}")
            
            if grid_search.best_score_ > best_overall_score:
                best_overall_score = grid_search.best_score_
                best_overall_model = grid_search.best_estimator_
                best_overall_name = model_name
                
        except Exception as e:
            print(f"Error while training {model_name}: {e}")
            # Continue to the next model even if one fails
            continue

    print(f"\n=== GRID SEARCH COMPLETE ===")
    print(f"Best Model Found: {best_overall_name} with Accuracy: {best_overall_score:.4f}")
    
    return best_overall_model, best_overall_name

def save_best_model(model: Any, filepath: str) -> None:
    """
    Saves the trained model to the specified filepath using pickle.
    
    Args:
        model (Any): The trained Scikit-Learn model/pipeline.
        filepath (str): The destination path for the pickle file.
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Successfully saved best model to {filepath}")
    except Exception as e:
        print(f"Failed to save model to {filepath}: {e}")
        raise

def main() -> None:
    """
    Main orchestration function to run the model selection pipeline.
    """
    # 1. Load and Split into Train (1) and Test (1)
    X_train, X_test, y_train, y_test = load_and_split_data('../data/train.csv')
    
    # 2. Get Configurations
    models_config = get_model_pipelines()
    
    # 3. Perform Grid Search (Includes 5-fold CV)
    best_model, best_name = perform_grid_search(X_train, y_train, models_config)
    
    if best_model is not None:
        print("\n--- MODEL EVALUATION ---")
        
        # 4a. Check Train (1) Accuracy (Audit Requirement < 0.98)
        train_accuracy = accuracy_score(y_train, best_model.predict(X_train))
        print(f"TRAINING ACCURACY: {train_accuracy:.4f}")
        if train_accuracy < 0.98:
            print("Train accuracy is < 0.98 (No massive overfitting).")
        else:
            print("Model is overfitting (Train accuracy >= 0.98).")
            
        # 4b. Check Test (1) Accuracy (Using the split data you noticed!)
        test_1_accuracy = accuracy_score(y_test, best_model.predict(X_test))
        print(f"VALIDATION 'Test (1)' ACCURACY: {test_1_accuracy:.4f}")
        
        # 5. Save Model
        save_best_model(best_model, '../results/best_model.pkl')
    else:
        print("No model was successfully trained. Exiting without saving.")

if __name__ == "__main__":
    main()