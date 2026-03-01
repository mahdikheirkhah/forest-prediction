import pandas as pd
import numpy as np
import logging
from typing import Tuple, Union

# Set up logging for this module
logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering to the dataframe based on project hints.
    """
    df = df.copy()
    
    # 1. Project Hint 1: Distance to Hydrology (Pythagorean Theorem)
    df['Distance_To_Hydrology'] = np.sqrt(
        df['Horizontal_Distance_To_Hydrology']**2 + 
        df['Vertical_Distance_To_Hydrology']**2
    )
    
    # 2. Project Hint 2: Distance to Fire Points
    df['Fire_Road_Dist_Diff'] = (
        df['Horizontal_Distance_To_Fire_Points'] - 
        df['Horizontal_Distance_To_Roadways']
    )
    
    return df

def get_preprocessed_data(filepath: str) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    """
    Loads data, applies feature engineering, and dynamically splits X and y.
    """
    logger.info(f"Loading raw data from: {filepath}")
    df = pd.read_csv(filepath)
    
    logger.info("Applying feature engineering...")
    df = engineer_features(df)
    
    if 'Cover_Type' in df.columns:
        X = df.drop('Cover_Type', axis=1)
        y = df['Cover_Type']
        logger.info(f"Successfully split data into features (X) and target (y). Shape: X={X.shape}")
        return X, y
    else:
        logger.info(f"Target 'Cover_Type' not found. Returning feature matrix only. Shape: {df.shape}")
        return df