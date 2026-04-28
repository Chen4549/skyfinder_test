import os
import warnings

os.environ['TORCHVISION_USE_NVJPEG'] = '0' 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import sys
sys.path.insert(0,"/gpfs/data/shenlab/hc4549/skyfinder_test/fusion")

import pandas as pd

from tabpfn import TabPFNRegressor

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from feature_extract import extract

def extract_feature_pfn():
    feat_dataset = "/gpfs/data/shenlab/hc4549/skyfinder_test/imbalanced-regression/Skyfinder-dir/data/tabpfn.csv"
    reweight = "sqrt_inv"
    lds = True
    lds_ks = 5
    lds_sigma = 2
    fds = True
    fds_ks = 5
    fds_sigma = 2
    checkpoint = "/gpfs/data/shenlab/hc4549/ucla/imbalanced-regression/Skyfinder-dir/checkpoint/skyfinder_30_balanced_resnet50_DIR_BASIC_lds_gau_5_2.0_fds_gau_5_2.0_0_1_0.9_adam_l1_0.001_256/ckpt.best.pth.tar"

    tr_feat, tr_path, te_feat, te_path = extract(feat_dataset, reweight, lds, lds_ks, lds_sigma, fds, fds_ks, fds_sigma, checkpoint)

    return tr_feat, tr_path, te_feat, te_path

def pca_pfn(PCA_num=256):

    tr_feat, tr_path, te_feat, te_path = extract_feature_pfn()

    # 2. Format as DataFrames
    df_test_feats = pd.DataFrame(te_feat)
    df_test_feats['path'] = te_path

    df_train_feats = pd.DataFrame(tr_feat)
    df_train_feats['path'] = tr_path

    all_features = pd.concat([df_train_feats, df_test_feats], ignore_index=True)

    train_feat_data = all_features[all_features['path'].isin(tr_path)].drop('path', axis=1)
    test_feat_data = all_features[all_features['path'].isin(te_path)].drop('path', axis=1)

    print("")
    print("-"*50)
    print(f"Applying PCA to compress feature to: {PCA_num}")

    scaler = StandardScaler()
    pca = PCA(n_components=PCA_num)

    train_scaled = scaler.fit_transform(train_feat_data)
    pca.fit(train_scaled)

    tr_reduced = pca.transform(train_scaled)
    te_reduced = pca.transform(scaler.transform(test_feat_data))

    print("PCA compress completed.\n")

    cols = [str(i) for i in range(PCA_num)]
    df_tr_red = pd.DataFrame(tr_reduced, columns=cols); df_tr_red['path'] = tr_path
    df_te_red = pd.DataFrame(te_reduced, columns=cols); df_te_red['path'] = te_path

    df_reduced = pd.concat([df_tr_red, df_te_red], ignore_index=True)

    return df_reduced

def prep(df_reduced):
    # --- 1. LOAD & PREP ---
    df_full_meta = pd.read_csv('/gpfs/data/shenlab/hc4549/skyfinder_test/imbalanced-regression/Skyfinder-dir/data/skyfinder_30_balanced.csv')
    df_pfn = pd.read_csv("/gpfs/data/shenlab/hc4549/skyfinder_test/imbalanced-regression/Skyfinder-dir/data/tabpfn.csv")

    df_pfn = df_pfn.dropna(subset=['label'])

    df = pd.merge(df_pfn, df_reduced, on='path', how='inner')
    df.columns = df.columns.astype(str)

    # --- 2. CALCULATE LABEL FREQUENCIES (TRAIN ONLY) ---
    # Get counts for labels specifically in the training set
    global_train_counts = df_full_meta[df_full_meta['split'] == 'train']['label'].value_counts()

    def categorize_label(label_val):
        # count = train_counts.get(label_val, 0)
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

def inference_pfn(df):
    # --- 3. SPLIT DATA ---
    X = df.drop(["label", "camera", "path", "split", "category", "prediction"], axis=1)
    y = df['label']

    all_cols = X.columns.tolist()
    metadata = [c for c in all_cols if not c.isdigit()]
    pca_feats = [c for c in all_cols if c.isdigit()]

    # Sort the numerical strings so they appear as 0, 1, 2... and not 0, 1, 10, 100
    pca_feats.sort(key=int)

    print("-"*50)
    print(f"TabPFN input Features:")
    print(f"  Metadata: {metadata}")
    print(f"  PCA Components: {pca_feats[0]}, {pca_feats[1]}, ..., {pca_feats[-1]} ({len(pca_feats)} total)")
    print(f"Target variable name: {y.name}\n")

    train_mask = df['split'] == 'train'
    test_mask = df['split'] == 'test'

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Store categories for the test set specifically for evaluation
    test_categories = df.loc[test_mask, 'category']

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

    print("--- FINAL(RESNET50+TABPFN) EVALUATION RESULTS ---")
    print(f"Overall: {get_metrics(results_df)}")

    for cat in ['Many', 'Medium', 'Low']:
        subset = results_df[results_df['Category'] == cat]
        print(f"{cat}: {get_metrics(subset)}")

def main(PCA_num):
    df_reduced = pca_pfn(PCA_num)
    df = prep(df_reduced)
    inference_pfn(df)

if __name__ == '__main__':
    PCA_num = 256
    main(PCA_num)