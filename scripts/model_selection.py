import os
import pandas as pd
import pickle
import logging
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from preprocessing_feature_engineering import get_preprocessed_data

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_and_split_data(filepath: str, test_size: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    try:
        logging.info(f"Initiating data load and split for {filepath}...")
        X, y = get_preprocessed_data(filepath)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        logging.info(f"Data split successful. Training size: {X_train.shape[0]}, Validation size: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error during data loading and splitting: {e}")
        raise

def get_model_pipelines() -> Dict[str, Dict[str, Any]]:
    """
    Defines model pipelines and hyperparameter grids tuned to balance 
    high validation accuracy with the < 0.98 training accuracy audit constraint.
    """
    pipelines_and_grids = {
        'LogisticRegression': {
            'model': Pipeline([('imputer', SimpleImputer(strategy='median')),
                               ('scaler', StandardScaler()), 
                               ('clf', LogisticRegression(solver='saga', max_iter=1000, random_state=42))]),
            'params': {
                'clf__C': [0.01, 0.1, 0.2, 0.3, 0.5, 1.0],  # Regularization strength
                # 'clf__penalty': ['l2']       # Type of penalty
            }
        },
        'KNN': {
            'model': Pipeline([('imputer', SimpleImputer(strategy='median')),
                               ('scaler', StandardScaler()), 
                               ('clf', KNeighborsClassifier())]),
            'params': {
                'clf__n_neighbors': [21, 31, 51], # Number of neighbors
                'clf__weights': ['uniform']       # Weighting function
            }
        },
        'RandomForest': {
            'model': Pipeline([('imputer', SimpleImputer(strategy='median')),
                               ('scaler', StandardScaler()), 
                               ('clf', RandomForestClassifier(n_jobs=-1, random_state=42))]),
            'params': {
                'clf__n_estimators': [50, 80, 100, 200],  # Number of trees
                'clf__max_depth': [5, 10, 15, 20],       # Depth limit (to prevent overfitting)
                'clf__min_samples_leaf': [20, 50] # Min samples per leaf (forces generalization)
            }
        },
        'GradientBoosting': {
            'model': Pipeline([('imputer', SimpleImputer(strategy='median')),
                               ('scaler', StandardScaler()), 
                               ('clf', GradientBoostingClassifier(random_state=42))]),
            'params': {
                'clf__n_estimators': [50, 100],  
                'clf__learning_rate': [0.05, 0.1],    # Lowered to prevent fast memorization
                'clf__max_depth': [3, 5],             # Removed 8. Trees must be shallow!
                'clf__min_samples_leaf': [20, 50]    # Added the same safety net as Random Forest
            }
        },
        'SVM': {
            'model': Pipeline([('imputer', SimpleImputer(strategy='median')),
                               ('scaler', StandardScaler()), 
                               ('clf', SVC(random_state=42))]),
            'params': {
                'clf__C': [0.01, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],             # Margin hardness
                'clf__kernel': ['rbf']            # Transformation type
            }
        }
    }
    return pipelines_and_grids

def perform_grid_search(X_train: pd.DataFrame, y_train: pd.Series, models_config: Dict[str, Dict[str, Any]]) -> Tuple[Any, str]:
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_overall_model = None
    best_overall_score = 0.0
    best_overall_name = ""

    for model_name, config in models_config.items():
        logging.info(f"Starting Grid Search for {model_name}...")
        try:
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=cv_strategy,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            logging.info(f"[{model_name}] Best Parameters: {grid_search.best_params_}")
            logging.info(f"[{model_name}] Best CV Accuracy: {grid_search.best_score_:.4f}")
            
            if grid_search.best_score_ > best_overall_score:
                best_overall_score = grid_search.best_score_
                best_overall_model = grid_search.best_estimator_
                best_overall_name = model_name
                
        except Exception as e:
            logging.error(f"Error while training {model_name}: {e}")
            continue

    logging.info(f"=== GRID SEARCH COMPLETE ===")
    logging.info(f"Best Model Found: {best_overall_name} with Accuracy: {best_overall_score:.4f}")
    
    return best_overall_model, best_overall_name

def save_best_model(model: Any, filepath: str) -> None:
    try:
        # Check if directory exists, create if not
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Successfully saved best model to {filepath} (Overwrites if existed)")
    except Exception as e:
        logging.error(f"Failed to save model to {filepath}: {e}")
        raise

def main() -> None:
    logging.info("--- STARTING MODEL SELECTION PIPELINE ---")
    
    # POINTING TO REAL DATA
    X_train, X_test, y_train, y_test = load_and_split_data('../data/train.csv')
    models_config = get_model_pipelines()
    
    best_model, best_name = perform_grid_search(X_train, y_train, models_config)
    
    if best_model is not None:
        logging.info("--- MODEL EVALUATION ---")
        train_accuracy = accuracy_score(y_train, best_model.predict(X_train))
        logging.info(f"TRAINING ACCURACY: {train_accuracy:.4f}")
        
        if train_accuracy < 0.98:
            logging.info("✅ Train accuracy is < 0.98 (No massive overfitting).")
        else:
            logging.warning("❌ Model is overfitting (Train accuracy >= 0.98).")
            
        test_1_accuracy = accuracy_score(y_test, best_model.predict(X_test))
        logging.info(f"VALIDATION 'Test (1)' ACCURACY: {test_1_accuracy:.4f}")
        
        save_best_model(best_model, '../results/best_model.pkl')
    else:
        logging.error("No model was successfully trained. Exiting without saving.")

if __name__ == "__main__":
    main()