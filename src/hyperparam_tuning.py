# src/hyperparam_tuning.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, models_dir: Path):
    setup_logging()
    logging.info("Starting hyperparameter tuning for XGBoost...")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
    }

    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, random_state=42, use_label_encoder=False, eval_metric='mlogloss')

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    logging.info(f"GridSearchCV completed. Best parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_

    # Save the best model
    best_model_path = models_dir / "xgboost_best_model.pkl"
    joblib.dump(best_model, best_model_path)
    logging.info(f"Best XGBoost model saved as '{best_model_path}'.")

    return best_model

def main():
    setup_logging()

    # Define base directories
    base_dir = Path(__file__).parent.parent  # Assuming src is inside CreditRiskModelling/
    processed_data_dir = base_dir / "processed_data"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load train data
    train_test_dir = processed_data_dir / "train_test_split"
    try:
        X_train = pd.read_csv(train_test_dir / "X_train.csv")
        y_train = pd.read_csv(train_test_dir / "y_train.csv").squeeze()
        logging.info(f"Loaded training data from '{train_test_dir}'.")
    except Exception as e:
        logging.error(f"Error loading train-test split data: {e}")
        raise

    # Perform hyperparameter tuning
    tune_xgboost(X_train, y_train, models_dir)

    logging.info("Hyperparameter tuning completed.")

if __name__ == "__main__":
    main()
