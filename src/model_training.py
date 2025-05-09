# src/model_training.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def train_models(df_encoded: pd.DataFrame, models_dir: Path, processed_data_dir: Path):
    setup_logging()
    logging.info("Starting model training...")

    # Define features and target
    y = df_encoded['Approved_Flag']
    X = df_encoded.drop(['Approved_Flag'], axis=1)
    logging.info("Defined features and target.")

    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logging.info("Encoded target variable 'Approved_Flag'.")

    # Save the LabelEncoder
    label_encoder_path = models_dir / "label_encoder.pkl"
    joblib.dump(label_encoder, label_encoder_path)
    logging.info(f"LabelEncoder saved as '{label_encoder_path}'.")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logging.info(f"Data split into training and testing sets: {X_train.shape[0]} train, {X_test.shape[0]} test.")

    # Convert y_train and y_test to pandas Series for .to_csv()
    y_train = pd.Series(y_train, name='Approved_Flag')
    y_test = pd.Series(y_test, name='Approved_Flag')

    # Save the train-test split for evaluation
    train_test_dir = processed_data_dir / "train_test_split"
    train_test_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(train_test_dir / "X_train.csv", index=False)
    X_test.to_csv(train_test_dir / "X_test.csv", index=False)
    y_train.to_csv(train_test_dir / "y_train.csv", index=False)
    y_test.to_csv(train_test_dir / "y_test.csv", index=False)
    logging.info(f"Train-test split data saved in '{train_test_dir}'.")

    # Initialize models
    rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    dt_classifier = DecisionTreeClassifier(max_depth=20, min_samples_split=10, random_state=42)

    # Train Random Forest
    logging.info("Training Random Forest classifier...")
    rf_classifier.fit(X_train, y_train)
    rf_model_path = models_dir / "random_forest_model.pkl"
    joblib.dump(rf_classifier, rf_model_path)
    logging.info(f"Random Forest model trained and saved as '{rf_model_path}'.")

    # Train XGBoost
    logging.info("Training XGBoost classifier...")
    xgb_classifier.fit(X_train, y_train)
    xgb_model_path = models_dir / "xgboost_model.pkl"
    joblib.dump(xgb_classifier, xgb_model_path)
    logging.info(f"XGBoost model trained and saved as '{xgb_model_path}'.")

    # Train Decision Tree
    logging.info("Training Decision Tree classifier...")
    dt_classifier.fit(X_train, y_train)
    dt_model_path = models_dir / "decision_tree_model.pkl"
    joblib.dump(dt_classifier, dt_model_path)
    logging.info(f"Decision Tree model trained and saved as '{dt_model_path}'.")

    logging.info("Model training completed.")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Define base directories
    base_dir = Path(__file__).parent.parent  # Assuming src is inside CreditRiskModelling/
    processed_data_dir = base_dir / "processed_data"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load encoded data
    encoded_data_path = processed_data_dir / "encoded_data.csv"
    try:
        df_encoded = pd.read_csv(encoded_data_path)
        logging.info(f"Loaded encoded data from '{encoded_data_path}'.")
    except Exception as e:
        logging.error(f"Error loading encoded data: {e}")
        raise

    # Train models
    train_models(df_encoded, models_dir, processed_data_dir)
