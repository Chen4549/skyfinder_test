import os
import warnings

os.environ['TORCHVISION_USE_NVJPEG'] = '0' 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import pandas as pd

from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prep():
    # --- 1. LOAD & PREP ---
    df_full_meta = pd.read_csv('./imbalanced-regression/Skyfinder-dir/data/skyfinder_30_balanced.csv')
    df = pd.read_csv("./imbalanced-regression/Skyfinder-dir/data/tabpfn.csv")

    df = df.dropna(subset=['label'])

    # --- 2. CALCULATE LABEL FREQUENCIES (TRAIN ONLY) ---
    global_train_counts = df_full_meta[df_full_meta['split'] == 'train']['label'].value_counts()

    def categorize_label(label_val):
        count = global_train_counts.get(label_val, 0)
        if count > 1000:
            return 'Many'
        elif 100 <= count <= 1000:
            return 'Medium'
        else:
            return 'Low'

    # Apply the categorization to the entire dataframe
    df['category'] = df['label'].apply(categorize_label)

    return df

def regression_pfn(df):
    # --- 3. SPLIT DATA ---
    X = df.drop(["label", "camera", "path", "split", "category", "prediction"], axis=1)
    y = df['label']

    train_mask = df['split'] == 'train'
    test_mask = df['split'] == 'test'

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Store categories for the test set specifically for evaluation
    test_categories = df.loc[test_mask, 'category']

    # TabPFN Sampling Constraint
    # if len(X_train) > 10000:
    #     X_train = X_train.sample(10000, random_state=42)
    #     y_train = y_train.loc[X_train.index]

    print(f"TabPFN Input features: {X.columns.tolist()}")
    print(f"Target variable name: {y.name}\n")

    # --- 4. TRAIN AND PREDICT ---
    reg = TabPFNRegressor(device='cuda') 
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)

    # --- 5. GROUPED EVALUATION ---
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions,
        'Category': test_categories
    })

    def get_metrics(df_subset):
        if len(df_subset) == 0: return "N/A"
        mse = mean_squared_error(df_subset['Actual'], df_subset['Predicted'])
        mae = mean_absolute_error(df_subset['Actual'], df_subset['Predicted'])
        return f"MSE: {mse:.4f} | MAE: {mae:.4f} | Count: {len(df_subset)}"

    print("--- TabPFN EVALUATION RESULTS ---")
    print(f"Overall: {get_metrics(results_df)}")

    for cat in ['Many', 'Medium', 'Low']:
        subset = results_df[results_df['Category'] == cat]
        print(f"{cat}: {get_metrics(subset)}")

def main():
    df = prep()
    regression_pfn(df)

if __name__ == '__main__':
    main()