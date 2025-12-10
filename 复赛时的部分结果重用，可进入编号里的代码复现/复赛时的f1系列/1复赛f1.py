import pandas as pd
import numpy as np
import os
import warnings
import joblib
from tqdm import tqdm

# Core ML/DS libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import lightgbm as lgb

warnings.filterwarnings('ignore')
tqdm.pandas()

# --- Configuration ---
class Config:
    DATA_FILE = '1data.csv'
    OUTPUT_FILE = '1复赛时的f1.csv'
    # Cache paths
    TEXT_PIPELINE_PATH = '1复赛时的text_pipeline_reborrow_final.joblib'
    MODEL_SAVE_PATH = '1复赛时的lgb_model_reborrow_final.joblib'
    FEATURE_LIST_PATH = '1复赛时的lgb_features_reborrow_final.pkl'
    # Feature Engineering Parameters
    TEXT_EMBEDDING_DIMS = 128

    # Model Parameters
    LGBM_ITERATIONS = 2000
    EARLY_STOPPING_ROUNDS = 150

# --- 1. 为复借任务构建训练数据 (保持不变) ---
def create_reborrow_training_data(file_path):
    print("Step 1: Loading data and creating training set for re-borrow task...")
    df = pd.read_csv(file_path)
    df['借阅时间'] = pd.to_datetime(df['借阅时间'], errors='coerce')
    df = df.dropna(subset=['借阅时间', 'user_id', 'book_id'])

    df['borrow_count'] = df.groupby(['user_id', 'book_id'])['inter_id'].transform('count')
    df['will_reborrow'] = (df['borrow_count'] > 1).astype(int)

    training_df = df.loc[df.groupby(['user_id', 'book_id'])['借阅时间'].idxmin()].copy()

    print(f"  - Created training set with {len(training_df)} unique user-book pairs.")

    cat_cols = ['gender', 'department', 'grade', 'user_type', '出版社', '一级分类', '二级分类']
    for col in cat_cols:
        if col in training_df.columns:
            training_df[col] = training_df[col].astype(str).fillna('missing')

    return df, training_df


# --- 2. 最终版特征工程：100%无泄漏 ---

# 2a. 内容特征 (安全)
def generate_content_embeddings(df, config):
    print("Step 2a: Generating content embeddings (TF-IDF + SVD)...")
    if os.path.exists(config.TEXT_PIPELINE_PATH):
        text_pipeline = joblib.load(config.TEXT_PIPELINE_PATH)
    else:
        text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1, 2))),
            ('svd', TruncatedSVD(n_components=config.TEXT_EMBEDDING_DIMS, random_state=42))
        ])

    text_cols = ['题名', '一级分类', '二级分类', '作者', '出版社']
    df['text_features_raw'] = df[text_cols].astype(str).fillna('missing').agg(' '.join, axis=1)

    if not os.path.exists(config.TEXT_PIPELINE_PATH):
        print("  - Fitting new text pipeline...")
        # 从去重的书籍元数据中拟合，更高效且安全
        book_texts = df[['book_id'] + text_cols].drop_duplicates(subset=['book_id'])
        book_texts_agg = book_texts[text_cols].astype(str).fillna('missing').agg(' '.join, axis=1)
        text_pipeline.fit(book_texts_agg)
        joblib.dump(text_pipeline, config.TEXT_PIPELINE_PATH)

    content_embeds = text_pipeline.transform(df['text_features_raw'])
    content_df = pd.DataFrame(
        content_embeds,
        index=df.index,
        columns=[f'content_embed_{i}' for i in range(config.TEXT_EMBEDDING_DIMS)]
    )
    return pd.concat([df.drop(columns=['text_features_raw']), content_df], axis=1)


# 2b. 即时行为特征 (安全)
def generate_instant_features(df):
    print("Step 2b: Generating instant (leak-free) features...")
    # 这两个特征是样本本身自带的，完全没有“看到”未来，是100%安全的
    df['first_borrow_renewed'] = df['是否续借'].astype(int)
    df['first_borrow_duration'] = df['总借阅时间(天)']

    df['first_borrow_duration'].fillna(0, inplace=True)
    return df


# --- 3. 模型训练 (保持不变) ---
def train_lgbm_model(df, config):
    print("Step 3: Training LightGBM model...")

    y = df['will_reborrow']
    # 移除与target直接相关或无用的列
    ignore_cols = ['will_reborrow', 'borrow_count', 'inter_id', 'user_id', 'book_id',
                   '借阅时间', '还书时间', '续借时间', '题名', '作者', '是否续借', '总借阅时间(天)']
    features = [col for col in df.columns if col not in ignore_cols]

    categorical_features = ['gender', 'department', 'grade', 'user_type', '出版社', '一级分类', '二级分类']
    for col in categorical_features:
        if col in df.columns:
            df[col] = pd.Categorical(df[col])

    X_train, X_val, y_train, y_val = train_test_split(df[features], y, test_size=0.2, random_state=42, stratify=y)

    print(f"  - Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=config.LGBM_ITERATIONS,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=True)],
        categorical_feature=categorical_features
    )

    joblib.dump(model, config.MODEL_SAVE_PATH)
    joblib.dump(features, config.FEATURE_LIST_PATH)
    return model, features


# --- 4. 推荐生成 (流程简化) ---
def generate_reborrow_recommendations(full_df_with_counts, model, features, config):
    print("Step 4: Generating re-borrow recommendations...")

    candidates = full_df_with_counts[full_df_with_counts['borrow_count'] == 1].copy()
    candidates = candidates.loc[candidates.groupby(['user_id', 'book_id'])['借阅时间'].idxmin()]

    print(f"Found {len(candidates)} unique user-book pairs with single borrow as candidates.")

    if candidates.empty: return

    print("  - Applying feature engineering to candidates...")

    # 只应用100%安全的特征
    featured_candidates = generate_content_embeddings(candidates, config)
    featured_candidates = generate_instant_features(featured_candidates)

    # 确保特征列对齐
    for col in features:
        if col not in featured_candidates.columns:
            featured_candidates[col] = 0

    categorical_features = ['gender', 'department', 'grade', 'user_type', '出版社', '一级分类', '二级分类']
    for col in categorical_features:
        if col in featured_candidates.columns:
            featured_candidates[col] = pd.Categorical(featured_candidates[col])

    print("  - Predicting re-borrow scores...")
    scores = model.predict_proba(featured_candidates[features])[:, 1]
    featured_candidates['score'] = scores

    recommendations_idx = featured_candidates.groupby('user_id')['score'].idxmax()
    recommendations = featured_candidates.loc[recommendations_idx]

    output_df = recommendations[['user_id', 'book_id', 'score']]
    output_df.to_csv(config.OUTPUT_FILE, index=False)
    print(f"Recommendations saved to {config.OUTPUT_FILE}. Shape: {output_df.shape}")


# --- 主执行流程 (最终版) ---
if __name__ == '__main__':
    cfg = Config()

    # 1. 加载数据
    full_df, training_df = create_reborrow_training_data(cfg.DATA_FILE)

    # 2. 应用100%无泄漏的特征工程
    featured_df = generate_content_embeddings(training_df, cfg)
    featured_df = generate_instant_features(featured_df)

    # 3. 训练模型
    model, feature_list = train_lgbm_model(featured_df, cfg)

    # 4. 生成推荐
    generate_reborrow_recommendations(full_df, model, feature_list, cfg)

    print("\nProcess finished successfully.")