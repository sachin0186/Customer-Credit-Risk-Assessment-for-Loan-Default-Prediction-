# src/model_evaluation.py

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import joblib
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model_path: Path, model_name: str, X_test: pd.DataFrame, y_test: pd.Series, label_encoder: joblib, evaluation_dir: Path):
    setup_logging()
    logging.info(f"Evaluating {model_name}...")

    # Load the model
    try:
        model = joblib.load(model_path)
        logging.info(f"Loaded {model_name} from '{model_path}'.")
    except Exception as e:
        logging.error(f"Error loading {model_name}: {e}")
        return

    # Make predictions
    y_pred_encoded = model.predict(X_test)

    # Decode predictions and true labels
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_true = label_encoder.inverse_transform(y_test)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=['P1', 'P2', 'P3', 'P4'])

    # Log the results
    logging.info(f"{model_name} - Accuracy: {accuracy:.2f}")
    for i, v in enumerate(['P1', 'P2', 'P3', 'P4']):
        logging.info(f"Class {v}: Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1 Score: {f1_score[i]:.2f}")

    # Save the classification report
    report = classification_report(y_true, y_pred, labels=['P1', 'P2', 'P3', 'P4'])
    report_path = evaluation_dir / f"{model_name}_classification_report.txt"
    with open(report_path, "w") as file:
        file.write(report)
    logging.info(f"Classification report saved as '{report_path}'.")

def main():
    setup_logging()

    # Define base directories
    base_dir = Path(__file__).parent.parent  # Assuming src is inside CreditRiskModelling/
    processed_data_dir = base_dir / "processed_data"
    models_dir = base_dir / "models"
    evaluation_dir = base_dir / "test_results" / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Load train-test split
    train_test_dir = processed_data_dir / "train_test_split"
    try:
        X_test = pd.read_csv(train_test_dir / "X_test.csv")
        y_test = pd.read_csv(train_test_dir / "y_test.csv").squeeze()
        logging.info(f"Loaded test data from '{train_test_dir}'.")
    except Exception as e:
        logging.error(f"Error loading train-test split data: {e}")
        raise

    # Load LabelEncoder
    label_encoder_path = models_dir / "label_encoder.pkl"
    try:
        label_encoder = joblib.load(label_encoder_path)
        logging.info(f"Loaded LabelEncoder from '{label_encoder_path}'.")
    except Exception as e:
        logging.error(f"Error loading LabelEncoder: {e}")
        raise

    # Define models to evaluate
    models = {
        "Random_Forest": models_dir / "random_forest_model.pkl",
        "XGBoost": models_dir / "xgboost_model.pkl",
        "Decision_Tree": models_dir / "decision_tree_model.pkl"
    }

    # Evaluate each model
    for model_name, model_path in models.items():
        evaluate_model(model_path, model_name, X_test, y_test, label_encoder, evaluation_dir)

    logging.info("Model evaluation completed.")

if __name__ == "__main__":
    main()
