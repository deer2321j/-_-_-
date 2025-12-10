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
OUTPUT_FILE = '旧决赛时的fv2.csv'
TOP10_OUTPUT_FILE = '旧决赛时的v2top10_recommendations.csv'  # 新增：top10推荐输出文件
MODEL_SAVE_PATH = '旧决赛时的fv2lgbm_recommender.joblib'
PREPROCESSOR_SAVE_PATH = '旧决赛时的fv2preprocessor_v2.joblib'
# --- Feature & Model Parameters ---
TEXT_EMBEDDING_DIMS = 128  # Dimensions for SVD text embeddings
# 定义特征列表（全局变量，便于各函数共享）
numerical_features = ['总借阅时间(天)', '续借延迟(天)', '续借次数']
categorical_features = ['gender', 'department', 'grade', 'user_type', '是否续借']


def preprocess_and_feature_engineer(file_path):
    """
    Loads data, creates the target variable, and engineers features using
    TF-IDF + SVD for text and appropriate encoders for other types.
    """
    print("Starting data preprocessing and feature engineering...")
    df = pd.read_csv(file_path)

    # 1. Create Target Variable 'will_re-borrow'
    df['borrow_count'] = df.groupby(['user_id', 'book_id'])['inter_id'].transform('count')
    df['will_re-borrow'] = (df['borrow_count'] > 1).astype(int)

    # 转换'借阅时间'为datetime类型，以便正确排序和使用idxmin
    df['借阅时间'] = pd.to_datetime(df['借阅时间'], errors='coerce')

    # 只使用第一次交互记录来预测是否会有第二次借阅
    df_first_interaction = df.loc[df.groupby(['user_id', 'book_id'])['借阅时间'].idxmin()].copy()

    print("Target variable created. Using first-interaction data for training.")

    # 2. Feature Engineering
    # 处理文本特征
    text_cols = ['题名', '一级分类', '二级分类', '作者', '出版社']
    for col in text_cols:
        df_first_interaction[col] = df_first_interaction[col].astype(str).fillna('missing')
    df_first_interaction['text_features'] = df_first_interaction[text_cols].agg(' '.join, axis=1)

    # 处理缺失值
    for col in numerical_features:
        df_first_interaction[col] = pd.to_numeric(df_first_interaction[col], errors='coerce').fillna(0)
    for col in categorical_features:
        df_first_interaction[col] = df_first_interaction[col].astype(str).fillna('Unknown')
        # 应用标签编码
        le = LabelEncoder()
        df_first_interaction[col] = le.fit_transform(df_first_interaction[col])

    # 3. 创建统一的特征集
    X = df_first_interaction[numerical_features + categorical_features].copy()
    y = df_first_interaction['will_re-borrow'].values

    # 4. 单独处理文本特征：TF-IDF + SVD
    print("Generating text embeddings with TF-IDF and SVD...")
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('svd', TruncatedSVD(n_components=TEXT_EMBEDDING_DIMS, random_state=42))
    ])

    text_embeddings = text_pipeline.fit_transform(df_first_interaction['text_features'])

    # 合并表格特征和文本嵌入
    X_final = np.hstack([X.values, text_embeddings])

    # 保存文本预处理管道
    joblib.dump(text_pipeline, PREPROCESSOR_SAVE_PATH)
    print(f"Text preprocessor saved to {PREPROCESSOR_SAVE_PATH}")

    return X_final, y, df_first_interaction


def train_lightgbm_model(X, y, user_ids_for_split=None, n_estimators=1000, retrain=False):
    """
    Trains or retrains a LightGBM model.
    If `retrain` is False, it performs cross-validation to find the best number of rounds.
    If `retrain` is True, it trains on the full dataset for a specified number of rounds.
    """
    scaler = StandardScaler()

    if not retrain:
        print("Starting LightGBM model training with cross-validation to find best iteration...")
        train_users, val_users = train_test_split(user_ids_for_split.unique(), test_size=0.2, random_state=42)

        train_indices = user_ids_for_split.isin(train_users)
        val_indices = user_ids_for_split.isin(val_users)

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        X_train[:, :len(numerical_features)] = scaler.fit_transform(X_train[:, :len(numerical_features)])
        X_val[:, :len(numerical_features)] = scaler.transform(X_val[:, :len(numerical_features)])

        lgbm = lgb.LGBMClassifier(objective='binary',
                                  metric='auc',
                                  random_state=42,
                                  n_estimators=1000,  # Will be pruned by early stopping
                                  learning_rate=0.05,
                                  num_leaves=31)

        lgbm.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 eval_metric='auc',
                 callbacks=[lgb.early_stopping(100, verbose=False)])

        print(f"Best iteration found: {lgbm.best_iteration_}")
        return lgbm.best_iteration_, scaler
    else:
        print(f"Retraining LightGBM model on the full dataset for {n_estimators} rounds...")
        X[:, :len(numerical_features)] = scaler.fit_transform(X[:, :len(numerical_features)])

        lgbm = lgb.LGBMClassifier(objective='binary',
                                  metric='auc',
                                  random_state=42,
                                  n_estimators=n_estimators,
                                  learning_rate=0.05,
                                  num_leaves=31)
        lgbm.fit(X, y)

        model_payload = {'model': lgbm, 'scaler': scaler}
        joblib.dump(model_payload, MODEL_SAVE_PATH)
        print(f"Retrained model and scaler saved to {MODEL_SAVE_PATH}")
        return lgbm, scaler


def generate_recommendations(original_df):
    """
    Generates book recommendations for each user based on the trained model.
    Now includes both top1 recommendation and top10 recommendations with probabilities.
    """
    print("Loading model and preprocessors to generate recommendations...")

    # 加载训练好的模型、标准化器和文本处理管道
    model_payload = joblib.load(MODEL_SAVE_PATH)
    model = model_payload['model']
    scaler = model_payload['scaler']
    text_pipeline = joblib.load(PREPROCESSOR_SAVE_PATH)

    # 确定候选书籍（只借过一次的）
    candidates_df = original_df[original_df['borrow_count'] == 1].copy()

    recommendations = []
    top10_recommendations = []  # 新增：存储top10推荐

    # 按用户分组以高效预测
    user_groups = candidates_df.groupby('user_id')

    print("Predicting recommendations for each user...")
    for user_id, group in tqdm(user_groups, desc="Processing users"):
        if group.empty:
            continue

        # 应用与训练时相同的特征工程
        # 文本特征
        text_cols = ['题名', '一级分类', '二级分类', '作者', '出版社']
        for col in text_cols:
            group[col] = group[col].astype(str).fillna('missing')
        group['text_features'] = group[text_cols].agg(' '.join, axis=1)

        # 处理数值和分类特征
        for col in numerical_features:
            group[col] = pd.to_numeric(group[col], errors='coerce').fillna(0)

        # 为分类特征创建一个全局编码器字典
        label_encoders = {}
        for col in categorical_features:
            group[col] = group[col].astype(str).fillna('Unknown')
            # 检查是否已拟合编码器，如果没有则拟合
            if col not in label_encoders:
                le = LabelEncoder()
                # 拟合时包含"Unknown"类别
                all_categories = pd.unique(group[col].tolist() + ['Unknown'])
                le.fit(all_categories)
                label_encoders[col] = le
            group[col] = label_encoders[col].transform(group[col])

        # 准备特征矩阵
        X_user_tabular = group[numerical_features + categorical_features].values
        X_user_text = text_pipeline.transform(group['text_features'])

        # 标准化数值特征
        X_user_tabular[:, :len(numerical_features)] = scaler.transform(X_user_tabular[:, :len(numerical_features)])

        X_user_final = np.hstack([X_user_tabular, X_user_text])

        # 获取预测分数（属于类别1的概率）
        scores = model.predict_proba(X_user_final)[:, 1]

        # 找到分数最高的书籍（原有功能）
        best_book_idx = np.argmax(scores)
        recommended_book_id = group['book_id'].iloc[best_book_idx]
        recommendations.append({'user_id': user_id, 'book_id': recommended_book_id})

        # 新增：获取top10推荐
        # 获取排序后的索引（从高到低）
        top_indices = np.argsort(scores)[::-1]
        # 取前10个，或者如果不足10个则取全部
        top_n = min(10, len(top_indices))
        top_indices = top_indices[:top_n]

        # 收集top10推荐的信息
        for idx in top_indices:
            book_id = group['book_id'].iloc[idx]
            confidence = scores[idx]
            title = group['题名'].iloc[idx]
            author = group['作者'].iloc[idx]

            top10_recommendations.append({
                'user_id': user_id,
                'book_id': book_id,
                '置信概率': confidence,
                '书名': title,
                '作者': author
            })

    # 保存原有推荐
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Top1 recommendations saved to {OUTPUT_FILE}")

    # 新增：保存top10推荐
    top10_df = pd.DataFrame(top10_recommendations)
    top10_df.to_csv(TOP10_OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"Top10 recommendations saved to {TOP10_OUTPUT_FILE}")


if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at '{DATA_FILE}'")
    else:
        # 1. 数据预处理和特征工程
        X_final, y, df_first_interaction = preprocess_and_feature_engineer(DATA_FILE)

        # 2. Find the optimal number of training rounds
        best_iteration, _ = train_lightgbm_model(X_final, y, user_ids_for_split=df_first_interaction['user_id'])

        # 3. Retrain the model on the full dataset with adjusted rounds
        new_n_estimators = max(1, best_iteration)  # Ensure at least 1 round
        train_lightgbm_model(X_final, y, n_estimators=new_n_estimators, retrain=True)

        # 4. 生成最终推荐
        full_df = pd.read_csv(DATA_FILE)
        full_df['borrow_count'] = full_df.groupby(['user_id', 'book_id'])['inter_id'].transform('count')
        generate_recommendations(full_df)

        print("Process completed successfully.")
