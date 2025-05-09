# src/data_loading_and_preprocessing.py

from pathlib import Path
import pandas as pd
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(raw_data_dir: Path, processed_data_dir: Path):
    setup_logging()
    logging.info("Loading datasets...")

    # Define file paths
    file1_path = raw_data_dir / "bank_data_credit_risk_project.xlsx"
    file2_path = raw_data_dir / "cibil_data_credit_risk_project.xlsx"

    # Load the datasets
    try:
        df1 = pd.read_excel(file1_path)
        df2 = pd.read_excel(file2_path)
        logging.info("Datasets loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        raise

    # Remove placeholder values from df1
    initial_rows_df1 = df1.shape[0]
    df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]
    logging.info(f"Removed {initial_rows_df1 - df1.shape[0]} rows with placeholder 'Age_Oldest_TL' in df1.")

    # Identify and remove columns with more than 10,000 placeholder values (-99999) in df2
    columns_to_be_removed = [col for col in df2.columns if df2[col].eq(-99999).sum() > 10000]
    if columns_to_be_removed:
        df2 = df2.drop(columns=columns_to_be_removed)
        logging.info(f"Removed columns with excessive placeholder values: {columns_to_be_removed}")
    else:
        logging.info("No columns with more than 10,000 placeholder values found in df2.")

    # Remove rows with any remaining placeholder values (-99999) in df2
    initial_rows_df2 = df2.shape[0]
    df2 = df2.loc[(df2 != -99999).all(axis=1)]
    logging.info(f"Removed {initial_rows_df2 - df2.shape[0]} rows with placeholder values in df2.")

    # Merge the two dataframes on 'PROSPECTID' with an inner join
    df = pd.merge(df1, df2, how='inner', on='PROSPECTID')
    logging.info(f"Merged df1 and df2. Resulting dataframe has {df.shape[0]} rows and {df.shape[1]} columns.")

    # Save the preprocessed data
    preprocessed_data_path = processed_data_dir / "preprocessed_data.csv"
    df.to_csv(preprocessed_data_path, index=False)
    logging.info(f"Preprocessed data saved as '{preprocessed_data_path}'.")
    
    return df

if __name__ == "__main__":
    # Define base directories
    base_dir = Path(__file__).parent.parent  # Assuming src is inside CreditRiskModelling/
    raw_data_dir = base_dir / "raw_data"
    processed_data_dir = base_dir / "processed_data"
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Execute preprocessing
    load_and_preprocess_data(raw_data_dir, processed_data_dir)
