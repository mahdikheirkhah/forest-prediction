import pandas as pd
from sklearn.model_selection import train_test_split

def create_stratified_sample(input_filepath: str, output_filepath: str, sample_size: int = 5000) -> None:
    """
    Reads a large dataset, takes a stratified sample based on the target variable, 
    and saves it to a new CSV for fast testing.
    
    Args:
        input_filepath (str): Path to the massive original CSV.
        output_filepath (str): Path where the tiny sample CSV will be saved.
        sample_size (int): Exact number of rows you want in your sample.
    """
    print(f"Loading original data from {input_filepath}...")
    df = pd.read_csv(input_filepath)
    
    # We use train_test_split to grab a perfectly stratified chunk.
    # By setting stratify=df['Cover_Type'], we guarantee the miniature dataset 
    # has the exact same percentages of tree types as the massive dataset.
    _, sample_df = train_test_split(
        df, 
        test_size=sample_size, 
        stratify=df['Cover_Type'], 
        random_state=42
    )
    
    # Save the miniature dataset
    sample_df.to_csv(output_filepath, index=False)
    print(f"\n✅ Successfully created stratified sample of {len(sample_df)} rows!")
    print(f"Saved to: {output_filepath}")
    
    # Prove to the auditor that the distributions match perfectly
    print("\n--- Distribution Check ---")
    original_dist = (df['Cover_Type'].value_counts(normalize=True) * 100).round(1)
    sample_dist = (sample_df['Cover_Type'].value_counts(normalize=True) * 100).round(1)
    
    comparison_df = pd.DataFrame({
        'Original %': original_dist,
        'Sample %': sample_dist
    }).sort_index()
    print(comparison_df)

if __name__ == "__main__":
    # Create a 5,000 row miniature train dataset
    create_stratified_sample(
        input_filepath='../data/train.csv', 
        output_filepath='../data/train_sample.csv', 
        sample_size=5000
    )
    
    # Create a 500 row miniature test dataset (using your test.csv which also has Cover_Type)
    create_stratified_sample(
        input_filepath='../data/test.csv', 
        output_filepath='../data/test_sample.csv', 
        sample_size=500
    )