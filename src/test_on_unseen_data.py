# src/test_on_unseen_data.py

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_unseen_data(unseen_file_path: Path, encoder_columns: pd.Index, scaler: StandardScaler):
    setup_logging()
    logging.info("Loading unseen data...")

    # Load unseen data
    try:
        unseen_df = pd.read_excel(unseen_file_path)
        logging.info(f"Unseen data loaded with {unseen_df.shape[0]} rows and {unseen_df.shape[1]} columns.")
    except Exception as e:
        logging.error(f"Error loading unseen data: {e}")
        raise

    # Remove placeholder values as done in preprocessing
    initial_rows = unseen_df.shape[0]
    unseen_df = unseen_df.loc[unseen_df['Age_Oldest_TL'] != -99999]
    rows_removed = initial_rows - unseen_df.shape[0]
    if rows_removed > 0:
        logging.info(f"Removed {rows_removed} rows with placeholder 'Age_Oldest_TL'.")

    for col in unseen_df.columns:
        if unseen_df[col].dtype != 'object':
            initial_col_rows = unseen_df.shape[0]
            unseen_df = unseen_df.loc[unseen_df[col] != -99999]
            col_rows_removed = initial_col_rows - unseen_df.shape[0]
            if col_rows_removed > 0:
                logging.info(f"Removed {col_rows_removed} rows with placeholder values in column '{col}'.")

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
    unseen_df['EDUCATION'] = unseen_df['EDUCATION'].map(education_map)
    unseen_df['EDUCATION'] = unseen_df['EDUCATION'].fillna(0).astype(int)
    logging.info("Ordinal encoding applied to 'EDUCATION'.")

    # One-hot encoding for categorical features
    categorical_cols = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
    unseen_df = pd.get_dummies(unseen_df, columns=categorical_cols)
    logging.info(f"One-hot encoding applied to {categorical_cols}.")

    # Align the unseen data with the training data columns
    unseen_df_encoded = unseen_df.reindex(columns=encoder_columns, fill_value=0)
    logging.info("Unseen data aligned with training data columns.")

    # Scale the numerical features
    columns_to_scale = [
        'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment',
        'max_recent_level_of_deliq', 'recent_level_of_deliq',
        'time_since_recent_enq', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr'
    ]
    try:
        unseen_df_encoded[columns_to_scale] = scaler.transform(unseen_df_encoded[columns_to_scale])
        logging.info("Numerical features scaled.")
    except Exception as e:
        logging.error(f"Error scaling numerical features: {e}")
        raise

    return unseen_df_encoded

def main():
    setup_logging()

    # Define base directories
    base_dir = Path(__file__).parent.parent  # Assuming src is inside CreditRiskModelling/
    raw_data_dir = base_dir / "raw_data"
    processed_data_dir = base_dir / "processed_data"
    models_dir = base_dir / "models"
    test_results_dir = base_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    # Path to the unseen data file
    unseen_file_path = base_dir / "testing_data" / "Unseen_Dataset.xlsx"  # Ensure the path is correct

    # Load the encoder columns from encoded_data.csv
    encoded_data_path = processed_data_dir / "encoded_data.csv"
    try:
        df_encoded = pd.read_csv(encoded_data_path)
        encoder_columns = df_encoded.drop(['Approved_Flag'], axis=1).columns
        logging.info(f"Loaded encoded data from '{encoded_data_path}'.")
    except Exception as e:
        logging.error(f"Error loading encoded data: {e}")
        raise

    # Load the scaler
    scaler_path = processed_data_dir / "scaler.pkl"
    try:
        scaler = joblib.load(scaler_path)
        logging.info(f"Loaded scaler from '{scaler_path}'.")
    except Exception as e:
        logging.error(f"Error loading scaler: {e}")
        raise

    # Preprocess unseen data
    unseen_df_processed = preprocess_unseen_data(unseen_file_path, encoder_columns, scaler)

    # Load the best XGBoost model
    best_xgb_model_path = models_dir / "xgboost_best_model.pkl"
    try:
        best_xgb_model = joblib.load(best_xgb_model_path)
        logging.info(f"Loaded best XGBoost model from '{best_xgb_model_path}'.")
    except Exception as e:
        logging.error(f"Error loading best XGBoost model: {e}")
        raise

    # Load LabelEncoder
    label_encoder_path = models_dir / "label_encoder.pkl"
    try:
        label_encoder = joblib.load(label_encoder_path)
        logging.info(f"Loaded LabelEncoder from '{label_encoder_path}'.")
    except Exception as e:
        logging.error(f"Error loading LabelEncoder: {e}")
        raise

    # Make predictions
    unseen_predictions_encoded = best_xgb_model.predict(unseen_df_processed)
    logging.info("Predictions made on unseen data.")

    # Decode predictions
    unseen_predictions = label_encoder.inverse_transform(unseen_predictions_encoded)

    # Load the original unseen data to append predictions
    try:
        unseen_original = pd.read_excel(unseen_file_path)
    except Exception as e:
        logging.error(f"Error loading original unseen data: {e}")
        raise

    # Add predictions to the original unseen data
    unseen_original['Predicted_Risk'] = unseen_predictions
    predictions_output_path = test_results_dir / "unseen_data_with_predictions.xlsx"
    unseen_original.to_excel(predictions_output_path, index=False)
    logging.info(f"Predictions on unseen data saved as '{predictions_output_path}'.")

    # If actual labels are available in unseen data, evaluate performance
    if 'Approved_Flag' in unseen_original.columns:
        logging.info("Actual labels found in unseen data. Evaluating performance...")
        y_true = unseen_original['Approved_Flag']
        y_pred = unseen_original['Predicted_Risk']
        
        # Calculate metrics
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, labels=['P1', 'P2', 'P3', 'P4'])
        report_path = test_results_dir / "unseen_data_classification_report.txt"
        with open(report_path, "w") as file:
            file.write(report)
        logging.info(f"Classification report for unseen data saved as '{report_path}'.")
    else:
        logging.info("No actual labels found in unseen data. Only predictions are saved.")

if __name__ == "__main__":
    main()
