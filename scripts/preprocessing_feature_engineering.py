import pandas as pd
import numpy as np
from typing import Tuple, Union

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering to the dataframe based on project hints.
    
    Args:
        df (pd.DataFrame): The raw dataframe.
        
    Returns:
        pd.DataFrame: The dataframe with new engineered features.
    """
    df = df.copy()
    
    # 1. Project Hint 1: Distance to Hydrology (Pythagorean Theorem)
    # sqrt((Horizontal_Distance_To_Hydrology)^2 + (Vertical_Distance_To_Hydrology)^2)
    df['Distance_To_Hydrology'] = np.sqrt(
        df['Horizontal_Distance_To_Hydrology']**2 + 
        df['Vertical_Distance_To_Hydrology']**2
    )
    
    # 2. Project Hint 2: Distance to Fire Points (Pythagorean Theorem)
    df['Fire_Road_Dist_Diff'] = (
        df['Horizontal_Distance_To_Fire_Points'] - 
        df['Horizontal_Distance_To_Roadways']
    )
    
    return df

def get_preprocessed_data(filepath: str) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    """
    Loads data, applies feature engineering, and dynamically splits X and y 
    if the target column exists.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        If 'Cover_Type' exists: Returns (X, y) where X is the feature matrix (unscaled) and y is the target.
        If 'Cover_Type' does NOT exist: Returns just the processed DataFrame (X, unscaled).
    """
    df = pd.read_csv(filepath)
    
    df = engineer_features(df)
    
    if 'Cover_Type' in df.columns:
        X = df.drop('Cover_Type', axis=1)
        y = df['Cover_Type']
        return X, y
    else:
        return df