# src/main.py

from pathlib import Path
import subprocess
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def run_script(script_path: Path):
    try:
        result = subprocess.run(['python', str(script_path)], check=True, capture_output=True, text=True)
        logging.info(f"Successfully executed {script_path.name}.")
        # Uncomment the next line to see the output of each script
        # logging.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing {script_path.name}: {e.stderr}")
        raise

def main():
    setup_logging()
    logging.info("Starting Credit Risk Analysis Pipeline...")

    # Define base directories
    base_dir = Path(__file__).parent.parent  # Assuming src is inside CreditRiskModelling/
    src_dir = base_dir / "src"
    scripts = [
        src_dir / "data_loading_and_preprocessing.py",
        src_dir / "feature_engineering.py",
        src_dir / "encoding_and_scaling.py",
        src_dir / "model_training.py",
        src_dir / "model_evaluation.py",
        src_dir / "hyperparam_tuning.py",
        src_dir / "test_on_unseen_data.py"
    ]

    # Execute each script in order
    for script in scripts:
        logging.info(f"Running script: {script.name}")
        run_script(script)

    logging.info("Credit Risk Analysis Pipeline completed successfully.")

if __name__ == "__main__":
    main()
