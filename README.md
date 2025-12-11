# Capstone: Credit Card Fraud EDA and Baseline Model

This repository contains a Jupyter notebook, `capstone_fraud_eda_report.ipynb`, that performs exploratory data analysis (EDA) and builds a baseline machine learning model for credit card fraud detection. The goal is to understand patterns in transaction data and create an initial classifier that distinguishes fraudulent from non‑fraudulent transactions.

## Dataset

The notebook expects a CSV file (for example, `output.csv`) with the following columns:

- `distance_from_home`: Distance from the cardholder’s home to the transaction location  
- `distance_from_last_transaction`: Distance between the current and previous transaction  
- `ratio_to_median_purchase_price`: Ratio of the transaction amount to the cardholder’s median purchase price  
- `repeat_retailer`: 1 if the retailer has been used before, otherwise 0  
- `used_chip`: 1 if a chip card was used, otherwise 0  
- `used_pin_number`: 1 if a PIN was used, otherwise 0  
- `online_order`: 1 if the transaction was online, otherwise 0  
- `fraud`: Target label (0 = non‑fraud, 1 = fraud)

## Notebook contents

The notebook is organized into the following sections:

   - Loads a sample of the data, converts `fraud` to integer, prints shape and fraud rate, and creates a `figures/` folder for plots.

2. **Data Checks and Cleaning**  
   - Prints dtypes and missing value counts.  
   - Computes descriptive statistics for numeric features.  
   - Examines upper quantiles of key continuous features.  
   - Optionally caps extreme outliers at the 99th percentile to reduce the impact of very large values.

3. **Exploratory Data Analysis (EDA)**  
   - **Target distribution:** Bar chart of fraud vs non‑fraud and overall fraud rate.  
   - **Univariate distributions:** Histograms and log‑scale histograms for distance and price ratio features.  
   - **Binary feature distributions:** Count plots for `repeat_retailer`, `used_chip`, `used_pin_number`, and `online_order`.  
   - **Feature vs fraud:**  
     - Boxplots (log‑scaled) of continuous features split by fraud label.  
     - KDE plots showing how distributions differ between fraud and non‑fraud.  
   - **Categorical vs fraud:** Proportion bar charts of binary features by fraud and non‑fraud.  
   - **Correlation analysis:**  
     - Correlation heatmap for all numeric features.  
     - Sorted correlations with `fraud` to highlight the most informative variables.  
   - **Pairwise relationships:** Pairplot on a sampled subset for the main continuous features, colored by fraud label.

4. **Feature Engineering**  
   - Builds a feature matrix `X` from the original columns and a target vector `y` from `fraud`.  
   - Adds log‑transformed versions of skewed continuous features:  
     - `log_distance_from_home`  
     - `log_distance_from_last_transaction`  
     - `log_ratio_to_median_purchase_price`  
   - Creates interaction features to capture online risk patterns:  
     - `online_and_far` = `online_order` × `log_distance_from_home`  
     - `high_ratio_online` = `online_order` × `log_ratio_to_median_purchase_price`

5. **Baseline Machine Learning Model**  
   - Splits the data into train and test sets with stratification on `fraud`.  
   - Standardizes continuous and log‑transformed features using `StandardScaler`.  
   - Trains a logistic regression model with `class_weight="balanced"` to address class imbalance.  
   - Evaluates the model with:  
     - Classification report (precision, recall, F1)  
     - ROC AUC score  
     - Confusion matrix

6. **Summary and Next Steps**  
   - Summarizes key insights from the EDA (for example, fraud tends to occur at larger distances, with higher purchase ratios, more often online, and less often with chip/PIN).  
   - Suggests improvements such as trying tree‑based models, tuning thresholds for better recall on fraud, and exploring additional or time‑based features.

## How to Run

1. Place your transaction CSV (e.g., `output.csv`) in a location accessible to the notebook.  
2. Open `capstone_fraud_eda_report.ipynb` in Jupyter, VS Code, or Google Colab.  
3. Update the `file_path` variable in the first code cell to point to your CSV.  
4. Run all cells from top to bottom. Plots will be saved under the `figures/` directory, and model metrics will be printed in the output.

## Requirements

Key Python packages used in the notebook include:

- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`

Install them with:

