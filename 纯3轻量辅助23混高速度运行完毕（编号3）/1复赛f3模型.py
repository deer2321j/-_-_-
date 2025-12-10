import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import warnings
import os
import joblib

# --- Configuration ---
warnings.filterwarnings('ignore')
DATA_FILE = '111data.csv'
# --- Output File Paths ---
TOP_RECOMMENDATIONS_OUTPUT_FILE = "olddata决赛时纯3final_recommendations纯3.csv"
TOP1_RECOMMENDATIONS_OUTPUT_FILE = '1olddata决赛时决赛纯3.csv'

# --- Specialized 3rd Borrow Model & Preprocessor Paths ---
MODEL_PATH = 'flgbm_3rd_borrow_model.joblib'
PREPROCESSOR_PATH = 'fpreprocessor_3rd_borrow.joblib'

# --- Feature & Model Parameters ---
TEXT_EMBEDDING_DIMS = 128
numerical_features = ['总借阅时间(天)', '续借延迟(天)', '续借次数']
categorical_features = ['gender', 'department', 'grade', 'user_type', '是否续借']
text_cols = ['题名', '一级分类', '二级分类', '作者', '出版社']


def prepare_base_data(file_path):
    """
    Loads the base data and performs initial processing like calculating borrow counts and ranks.
    """
    print("Loading and preparing base data...")
    df = pd.read_csv(file_path)
    df['借阅时间'] = pd.to_datetime(df['借阅时间'], errors='coerce')
    df['还书时间'] = pd.to_datetime(df['还书时间'], errors='coerce')

    # Calculate borrow count and rank for each user-book pair
    df['borrow_count'] = df.groupby(['user_id', 'book_id'])['inter_id'].transform('count')
    df['borrow_rank'] = df.groupby(['user_id', 'book_id'])['借阅时间'].rank(method='first').astype(int)

    # Fill NaNs in key columns
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in categorical_features + text_cols:
        df[col] = df[col].astype(str).fillna('missing')

    return df


def prepare_data_for_3rd_borrow(df):
    """
    Prepares training data specifically for the 3rd borrow prediction model.
    Uses the second interaction's features to predict if a third borrow will occur.
    """
    print("\n--- Preparing data for 3rd Borrow Prediction Model ---")
    # Base for training: the second borrow instance
    df_second = df[df['borrow_rank'] == 2].copy()

    if df_second.empty:
        print("Warning: No second-borrow data found. Cannot train the 3rd borrow model.")
        return None, None, None

    # Target: Did a 3rd borrow happen?
    df_second['target'] = (df_second['borrow_count'] >= 3).astype(int)

    print(f"Found {len(df_second)} second-borrow instances for training.")
    print(f"Positive samples (3+ borrows): {df_second['target'].sum()}")

    # Feature Engineering
    df_second['text_features'] = df_second[text_cols].agg(' '.join, axis=1)

    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df_second[col] = le.fit_transform(df_second[col])
        label_encoders[col] = le

    X_tabular = df_second[numerical_features + categorical_features].values
    y = df_second['target'].values

    print("Generating text embeddings for the 3rd borrow model...")
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('svd', TruncatedSVD(n_components=TEXT_EMBEDDING_DIMS, random_state=42))
    ])
    text_embeddings = text_pipeline.fit_transform(df_second['text_features'])

    X_final = np.hstack([X_tabular, text_embeddings])

    preprocessor_payload = {'text_pipeline': text_pipeline, 'label_encoders': label_encoders}
    joblib.dump(preprocessor_payload, PREPROCESSOR_PATH)
    print(f"3rd borrow preprocessor saved to {PREPROCESSOR_PATH}")

    return X_final, y, df_second['user_id']


def train_3rd_borrow_model(X, y, user_ids_for_split, numerical_feature_count, model_save_path, n_estimators=1000,
                           retrain=False):
    """
    Trains or retrains the specialized LightGBM model for 3rd borrow prediction.
    """
    scaler = StandardScaler()

    if not retrain:
        print(f"Starting training for {model_save_path} with CV...")
        train_users, val_users = train_test_split(user_ids_for_split.unique(), test_size=0.2, random_state=42)
        train_indices = user_ids_for_split.isin(train_users)
        val_indices = user_ids_for_split.isin(val_users)

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        X_train[:, :numerical_feature_count] = scaler.fit_transform(X_train[:, :numerical_feature_count])
        X_val[:, :numerical_feature_count] = scaler.transform(X_val[:, :numerical_feature_count])

        lgbm = lgb.LGBMClassifier(objective='binary', metric='auc', random_state=42, n_estimators=1000,
                                  learning_rate=0.05, num_leaves=31)
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc',
                 callbacks=[lgb.early_stopping(100, verbose=False)])

        print(f"Best iteration found: {lgbm.best_iteration_}")
        return lgbm.best_iteration_, scaler
    else:
        print(f"Retraining {model_save_path} on full dataset for {n_estimators} rounds...")
        X[:, :numerical_feature_count] = scaler.fit_transform(X[:, :numerical_feature_count])
        lgbm = lgb.LGBMClassifier(objective='binary', metric='auc', random_state=42, n_estimators=n_estimators,
                                  learning_rate=0.05, num_leaves=31)
        lgbm.fit(X, y)

        model_payload = {'model': lgbm, 'scaler': scaler}
        joblib.dump(model_payload, model_save_path)
        print(f"Retrained 3rd borrow model and scaler saved to {model_save_path}")
        return lgbm, scaler


def generate_3rd_borrow_recommendations(df):
    """
    Generates 3rd borrow recommendations using the specialized model.
    The candidate set is strictly books that have been borrowed exactly twice.
    """
    print("\n--- Generating 3rd Borrow Recommendations ---")

    # Load the specialized model and preprocessor
    model_payload = joblib.load(MODEL_PATH)
    model = model_payload['model']
    scaler = model_payload['scaler']
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    text_pipeline = preprocessor['text_pipeline']
    le = preprocessor['label_encoders']

    # --- Candidate Set: Books borrowed exactly twice ---
    # We use the features from the 2nd borrow to predict the 3rd, matching the training data.
    candidates = df[(df['borrow_count'] == 2) & (df['borrow_rank'] == 2)].copy()

    if candidates.empty:
        print("No candidates found for 3rd borrow prediction (i.e., no books borrowed exactly twice).")
        return

    print(f"Found {len(candidates)} candidates for 3rd borrow prediction.")

    # --- Feature Preparation for Prediction ---
    candidates['text_features'] = candidates[text_cols].agg(' '.join, axis=1)
    for col in categorical_features:
        known_labels = le[col].classes_
        candidates[col] = candidates[col].apply(lambda x: x if x in known_labels else 'missing')
        candidates[col] = le[col].transform(candidates[col])

    X_tab = candidates[numerical_features + categorical_features].values
    X_text = text_pipeline.transform(candidates['text_features'])
    X_tab[:, :len(numerical_features)] = scaler.transform(X_tab[:, :len(numerical_features)])
    X_final = np.hstack([X_tab, X_text])

    # --- Prediction ---
    scores = model.predict_proba(X_final)[:, 1]

    final_recommendations = []
    for i, score in enumerate(scores):
        rec = candidates.iloc[i]
        final_recommendations.append({
            'user_id': rec['user_id'], 'book_id': rec['book_id'], '书名': rec['题名'],
            '作者': rec['作者'], '置信概率': score
        })

    # --- Finalize and Save ---
    print(f"\nGenerated {len(final_recommendations)} total 3rd borrow recommendations.")

    if not final_recommendations:
        print("No recommendations were generated.")
        return

    recs_df = pd.DataFrame(final_recommendations)
    recs_df_sorted = recs_df.sort_values('置信概率', ascending=False)

    top10_recs = recs_df_sorted.groupby('user_id').head(10)
    top10_recs.to_csv(TOP_RECOMMENDATIONS_OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nTop 10 recommendations saved to {TOP_RECOMMENDATIONS_OUTPUT_FILE}")

    top1_recs = recs_df_sorted.groupby('user_id').head(1)
    top1_recs.to_csv(TOP1_RECOMMENDATIONS_OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"Top 1 recommendations for submission saved to {TOP1_RECOMMENDATIONS_OUTPUT_FILE}")


if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at '{DATA_FILE}'")
    else:
        base_df = prepare_base_data(DATA_FILE)

        # --- Train the Specialized 3rd Borrow Model ---
        X, y, user_ids = prepare_data_for_3rd_borrow(base_df)

        if X is not None:
            best_iter, _ = train_3rd_borrow_model(X, y, user_ids, len(numerical_features), MODEL_PATH)
            train_3rd_borrow_model(X, y, user_ids, len(numerical_features), MODEL_PATH, n_estimators=max(1, best_iter),
                                   retrain=True)

            # --- Generate Recommendations with the New Model ---
            generate_3rd_borrow_recommendations(base_df)
            print("\nProcess completed successfully.")
        else:
            print("\nProcess finished, but no model was trained due to lack of data.")