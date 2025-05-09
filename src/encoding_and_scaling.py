# src/encoding_and_scaling.py

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def encode_and_scale(df: pd.DataFrame, processed_data_dir: Path, encoded_data_dir: Path):
    setup_logging()
    logging.info("Starting encoding and scaling...")
    
    # Ordinal encoding for 'EDUCATION'
    education_map = {
        'SSC': 1,
        '12TH': 2,
        'GRADUATE': 3,
        'UNDER GRADUATE': 3,
        'POST-GRADUATE': 4,
        'OTHERS': 1,
        'PROFESSIONAL': 3
    }
    df['EDUCATION'] = df['EDUCATION'].map(education_map)
    if df['EDUCATION'].isnull().any():
        logging.warning("Some 'EDUCATION' values were not mapped and are set to NaN.")
    df['EDUCATION'] = df['EDUCATION'].fillna(0).astype(int)
    logging.info("Ordinal encoding applied to 'EDUCATION'.")
    
    # One-hot encoding for other categorical variables
    categorical_cols = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    logging.info(f"One-hot encoding applied to {categorical_cols}.")
    
    # Columns to scale
    columns_to_scale = [
        'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment',
        'max_recent_level_of_deliq', 'recent_level_of_deliq',
        'time_since_recent_enq', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr'
    ]
    
    scaler = StandardScaler()
    df_encoded[columns_to_scale] = scaler.fit_transform(df_encoded[columns_to_scale])
    logging.info(f"Standard scaling applied to {columns_to_scale}.")
    
    # Save the scaler for unseen data
    scaler_path = encoded_data_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved as '{scaler_path}'.")
    
    # Save the encoded and scaled data
    encoded_data_path = encoded_data_dir / "encoded_data.csv"
    df_encoded.to_csv(encoded_data_path, index=False)
    logging.info(f"Encoding and scaling completed. Data saved as '{encoded_data_path}'.")
    
    return df_encoded

if __name__ == "__main__":
    # Define base directories
    base_dir = Path(__file__).parent.parent  # Assuming src is inside CreditRiskModelling/
    processed_data_dir = base_dir / "processed_data"
    encoded_data_dir = processed_data_dir  # Reuse processed_data for encoded data
    encoded_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load engineered data
    engineered_data_path = processed_data_dir / "engineered_data.csv"
    try:
        df_engineered = pd.read_csv(engineered_data_path)
        logging.info(f"Loaded engineered data from '{engineered_data_path}'.")
    except Exception as e:
        logging.error(f"Error loading engineered data: {e}")
        raise
    
    # Perform encoding and scaling
    encode_and_scale(df_engineered, processed_data_dir, encoded_data_dir)
