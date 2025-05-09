# src/feature_engineering.py

from pathlib import Path
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def feature_engineering(df: pd.DataFrame, processed_data_dir: Path):
    setup_logging()
    logging.info("Starting feature engineering...")

    # Define categorical features to test
    categorical_features = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
    significant_categorical = []

    logging.info("Performing Chi-Square tests on categorical features...")
    for feature in categorical_features:
        try:
            chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[feature], df['Approved_Flag']))
            logging.info(f"Chi-square test for '{feature}': p-value = {pval}")
            if pval <= 0.05:
                significant_categorical.append(feature)
                logging.info(f"'{feature}' is significant and retained.")
            else:
                logging.info(f"'{feature}' is not significant and removed.")
        except Exception as e:
            logging.error(f"Error performing Chi-Square test on '{feature}': {e}")
            continue

    # Identify numerical columns excluding identifiers and target
    numerical_columns = [col for col in df.columns if df[col].dtype != 'object' and col not in ['PROSPECTID', 'Approved_Flag']]
    logging.info(f"Numerical columns identified for VIF: {numerical_columns}")

    # Calculate VIF and remove multicollinear features
    vif_data = df[numerical_columns].copy()
    vif_threshold = 6
    columns_to_keep_vif = numerical_columns.copy()
    iteration = 1
    while True:
        vif = pd.Series([variance_inflation_factor(vif_data.values, i) 
                         for i in range(vif_data.shape[1])], index=vif_data.columns)
        max_vif = vif.max()
        if max_vif > vif_threshold:
            max_vif_col = vif.idxmax()
            logging.info(f"Iteration {iteration}: '{max_vif_col}' has VIF={max_vif}. Removing it.")
            vif_data = vif_data.drop(columns=[max_vif_col])
            columns_to_keep_vif.remove(max_vif_col)
            iteration += 1
        else:
            break

    logging.info(f"Columns retained after VIF check: {columns_to_keep_vif}")

    # Perform ANOVA for numerical features
    significant_numerical = []
    logging.info("Performing ANOVA tests on numerical features...")
    for feature in columns_to_keep_vif:
        try:
            groups = [group[feature].dropna() for name, group in df.groupby('Approved_Flag')]
            if len(groups) < 2:
                logging.warning(f"Not enough groups to perform ANOVA for '{feature}'. Skipping.")
                continue
            f_statistic, p_value = f_oneway(*groups)
            logging.info(f"ANOVA test for '{feature}': p-value = {p_value}")
            if p_value <= 0.05:
                significant_numerical.append(feature)
                logging.info(f"'{feature}' is significant and retained.")
            else:
                logging.info(f"'{feature}' is not significant and removed.")
        except Exception as e:
            logging.error(f"Error performing ANOVA test on '{feature}': {e}")
            continue

    # Combine significant features
    final_features = significant_numerical + significant_categorical
    logging.info(f"Final selected features: {final_features}")

    # Select the final features along with the target variable
    df_selected = df[final_features + ['Approved_Flag']]
    engineered_data_path = processed_data_dir / "engineered_data.csv"
    df_selected.to_csv(engineered_data_path, index=False)
    logging.info(f"Feature engineering completed. Data saved as '{engineered_data_path}'.")

    return df_selected

if __name__ == "__main__":
    # Define base directories
    base_dir = Path(__file__).parent.parent  
    processed_data_dir = base_dir / "processed_data"
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    preprocessed_data_path = processed_data_dir / "preprocessed_data.csv"
    try:
        df_preprocessed = pd.read_csv(preprocessed_data_path)
        logging.info(f"Loaded preprocessed data from '{preprocessed_data_path}'.")
    except Exception as e:
        logging.error(f"Error loading preprocessed data: {e}")
        raise

    # Perform feature engineering
    feature_engineering(df_preprocessed, processed_data_dir)
