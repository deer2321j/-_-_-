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
from catboost import CatBoostClassifier

# Graph-based feature libraries
import networkx as nx
from node2vec import Node2Vec

warnings.filterwarnings('ignore')
tqdm.pandas()


# --- Configuration ---
class Config:
    DATA_FILE = '111data.csv'
    OUTPUT_FILE = '256fv5.csv'
    TOP10_OUTPUT_FILE = 'ftop10_recommendations_v5.csv'  # 新增：top10推荐输出文件
    # Cache paths
    TEXT_PIPELINE_PATH = '1288ftext_pipeline_v5.joblib'
    GRAPH_EMBEDDINGS_PATH = '旧决赛时fgraph_embeddings_v5.pkl'
    MODEL_SAVE_PATH = 'fcatboost_model_v5.joblib'
    FEATURE_LIST_PATH = 'fcatboost_features_v5.pkl'

    # Feature Engineering Parameters
    TEXT_EMBEDDING_DIMS = 128
    GRAPH_EMBEDDING_DIMS = 64

    # Model Parameters
    CATBOOST_ITERATIONS = 1000
    EARLY_STOPPING_ROUNDS = 100


# --- 1. Data Loading & Initial Prep ---
def load_and_prepare_data(file_path):
    print("Step 1: Loading and preparing data...")
    df = pd.read_csv(file_path)

    # 检查并确保关键列存在
    required_cols = ['user_id', 'book_id', 'inter_id', '借阅时间']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据文件缺少必要列: {missing_cols}")

    # 转换“借阅时间”列为datetime类型，无法转换的设为NaT
    df['借阅时间'] = pd.to_datetime(df['借阅时间'], errors='coerce')
    # 删除借阅时间为NaT的行
    df = df.dropna(subset=['借阅时间'])

    # 初始化分类特征（提前转换为字符串避免后续类型问题）
    cat_cols = ['gender', 'department', 'grade', 'user_type', '出版社', '一级分类', '二级分类']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing')  # 缺失值用'missing'填充
        else:
            df[col] = 'missing'  # 若列不存在，统一填充为'missing'

    df['borrow_count'] = df.groupby(['user_id', 'book_id'])['inter_id'].transform('count')
    df['will_re-borrow'] = (df['borrow_count'] > 1).astype(int)

    # 提取首次交互数据
    df_model_data = df.loc[df.groupby(['user_id', 'book_id'])['借阅时间'].idxmin()].copy()
    print(f"Data prepared. Using {len(df_model_data)} first-interaction samples for modeling.")
    return df, df_model_data


# --- 2. Hybrid Feature Engineering ---

# 2a. Content Features (TF-IDF + SVD)
def generate_content_embeddings(df, config):
    print("Step 2a: Generating content embeddings (TF-IDF + SVD)...")
    # 保留关键ID列，避免后续特征工程中丢失
    keep_cols = ['user_id', 'book_id', 'inter_id']
    df = df[keep_cols + [col for col in df.columns if col not in keep_cols]]

    if os.path.exists(config.TEXT_PIPELINE_PATH):
        print("  - Loading cached text pipeline.")
        text_pipeline = joblib.load(config.TEXT_PIPELINE_PATH)
    else:
        print("  - Fitting new text pipeline...")
        text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1, 2))),
            ('svd', TruncatedSVD(n_components=config.TEXT_EMBEDDING_DIMS, random_state=42))
        ])

    text_cols = ['题名', '一级分类', '二级分类', '作者', '出版社']
    df['text_features_raw'] = df[text_cols].astype(str).fillna('missing').agg(' '.join, axis=1)

    if not os.path.exists(config.TEXT_PIPELINE_PATH):
        text_pipeline.fit(df['text_features_raw'])
        joblib.dump(text_pipeline, config.TEXT_PIPELINE_PATH)

    content_embeds = text_pipeline.transform(df['text_features_raw'])
    content_df = pd.DataFrame(content_embeds, index=df.index,
                              columns=[f'content_embed_{i}' for i in range(config.TEXT_EMBEDDING_DIMS)])
    return pd.concat([df, content_df], axis=1)


# 2b. Structural Features (Node2Vec)
def generate_graph_embeddings(full_df, config):
    print("Step 2b: Generating structural graph embeddings (Node2Vec)...")
    if os.path.exists(config.GRAPH_EMBEDDINGS_PATH):
        print("  - Loading cached graph embeddings.")
        return joblib.load(config.GRAPH_EMBEDDINGS_PATH)

    print("  - Building interaction graph...")
    # 构建用户-书籍交互图
    edges = full_df[['user_id', 'book_id']].apply(lambda row: (f"u_{row.user_id}", f"b_{row.book_id}"), axis=1).tolist()
    graph = nx.Graph(edges)

    print("  - Training Node2Vec model...")
    node2vec = Node2Vec(graph, dimensions=config.GRAPH_EMBEDDING_DIMS,
                        walk_length=15, num_walks=50,
                        workers=32, quiet=True)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    print("  - Extracting embeddings...")
    embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
    joblib.dump(embeddings, config.GRAPH_EMBEDDINGS_PATH)
    return embeddings


def merge_graph_embeddings(df, embeddings, config):
    # 保留关键ID列
    keep_cols = ['user_id', 'book_id', 'inter_id']
    df = df[keep_cols + [col for col in df.columns if col not in keep_cols]]

    user_embeds = df['user_id'].apply(lambda x: embeddings.get(f"u_{x}", np.zeros(config.GRAPH_EMBEDDING_DIMS)))
    book_embeds = df['book_id'].apply(lambda x: embeddings.get(f"b_{x}", np.zeros(config.GRAPH_EMBEDDING_DIMS)))

    user_embed_df = pd.DataFrame(user_embeds.tolist(), index=df.index,
                                 columns=[f'user_graph_embed_{i}' for i in range(config.GRAPH_EMBEDDING_DIMS)])
    book_embed_df = pd.DataFrame(book_embeds.tolist(), index=df.index,
                                 columns=[f'book_graph_embed_{i}' for i in range(config.GRAPH_EMBEDDING_DIMS)])

    return pd.concat([df, user_embed_df, book_embed_df], axis=1)


# 2c. Statistical Features
def generate_statistical_features(df):
    print("Step 2c: Generating statistical features...")
    # 保留关键ID列
    keep_cols = ['user_id', 'book_id', 'inter_id']
    df = df[keep_cols + [col for col in df.columns if col not in keep_cols]]

    # 用户统计特征
    user_stats = df.groupby('user_id').agg(
        user_total_borrows=('inter_id', 'count'),
        user_renewal_rate=('是否续借', 'mean')
    ).reset_index()
    df = pd.merge(df, user_stats, on='user_id', how='left')
    # 填充用户统计特征的缺失值
    df['user_total_borrows'] = df['user_total_borrows'].fillna(0)
    df['user_renewal_rate'] = df['user_renewal_rate'].fillna(0)

    # 书籍统计特征
    book_stats = df.groupby('book_id').agg(
        book_popularity=('inter_id', 'count'),
        book_unique_users=('user_id', 'nunique')
    ).reset_index()
    df = pd.merge(df, book_stats, on='book_id', how='left')
    # 填充书籍统计特征的缺失值
    df['book_popularity'] = df['book_popularity'].fillna(0)
    df['book_unique_users'] = df['book_unique_users'].fillna(0)

    return df


# --- 3. Model Training ---
def train_catboost_model(df, config):
    print("Step 3: Training CatBoost model...")

    # 保留关键ID列用于后续处理
    keep_cols = ['user_id', 'book_id']
    df = df[keep_cols + [col for col in df.columns if col not in keep_cols]]

    # 定义目标变量和特征
    y = df['will_re-borrow']
    ignore_cols = ['will_re-borrow', 'borrow_count', 'inter_id', 'user_id', 'book_id',
                   '借阅时间', '还书时间', '续借时间', '题名', '作者', 'text_features_raw']
    features = [col for col in df.columns if col not in ignore_cols]

    # 处理分类特征
    categorical_features = ['gender', 'department', 'grade', 'user_type', '出版社', '一级分类', '二级分类']
    existing_cat_features = [col for col in categorical_features if col in features]
    for col in existing_cat_features:
        df[col] = df[col].astype(str).fillna('missing')

    # 获取分类特征索引
    categorical_features_indices = [features.index(col) for col in existing_cat_features]

    # --- Find Optimal Iterations with Validation Set ---
    print("  - Finding optimal iterations with a validation set...")
    unique_users = df['user_id'].unique()
    train_users, val_users = train_test_split(unique_users, test_size=0.2, random_state=42)
    train_idx = df[df['user_id'].isin(train_users)].index
    val_idx = df[df['user_id'].isin(val_users)].index

    X_train, y_train = df.loc[train_idx, features], y.loc[train_idx]
    X_val, y_val = df.loc[val_idx, features], y.loc[val_idx]

    print(f"  - Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

    # 初始化CatBoost模型用于寻找最佳迭代次数
    temp_model = CatBoostClassifier(
        iterations=config.CATBOOST_ITERATIONS,
        learning_rate=0.05,
        depth=8,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=50,
        cat_features=categorical_features_indices
    )

    temp_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        use_best_model=True
    )

    best_iteration = temp_model.get_best_iteration() - 30
    if best_iteration is None:
        best_iteration = config.CATBOOST_ITERATIONS  # Fallback
        print(f"  - Early stopping did not trigger. Using max iterations: {best_iteration}")
    else:
        print(f"  - Optimal number of iterations found: {best_iteration}")

    # --- Train Final Model on Full Dataset ---
    print("  - Retraining final model on the entire dataset...")
    X_full, y_full = df[features], y

    final_model = CatBoostClassifier(
        iterations=best_iteration,  # 使用最佳迭代次数
        learning_rate=0.05,
        depth=8,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=50,
        cat_features=categorical_features_indices
    )

    final_model.fit(X_full, y_full)  # No eval_set, no early stopping

    print(f"Saving final model to {config.MODEL_SAVE_PATH}")
    joblib.dump(final_model, config.MODEL_SAVE_PATH)
    joblib.dump(features, config.FEATURE_LIST_PATH)
    return final_model, features


# --- 4. Recommendation Generation ---
# --- 4. Recommendation Generation ---
def generate_recommendations(full_df, model, features, config):
    print("Step 4: Generating recommendations...")

    # 候选集：只借过一次的书籍
    candidates = full_df[full_df['borrow_count'] == 1].copy()
    print(f"Found {len(candidates)} candidate interactions for recommendation.")
    if candidates.empty:
        print("No candidates available.")
        return

    # 确保user_id和book_id存在
    for col in ['user_id', 'book_id']:
        if col not in candidates.columns:
            raise KeyError(f"候选集中缺少必要列: {col}")

    # 为候选集生成特征
    print("  - Applying feature engineering to candidates...")
    candidates = generate_content_embeddings(candidates, config)
    graph_embeddings = joblib.load(config.GRAPH_EMBEDDINGS_PATH)
    candidates = merge_graph_embeddings(candidates, graph_embeddings, config)
    candidates = generate_statistical_features(candidates)

    # 确保所有特征存在且类型正确
    for col in features:
        if col not in candidates.columns:
            candidates[col] = 0  # 缺失特征填充0
        # 处理候选集中的分类特征
        if col in ['gender', 'department', 'grade', 'user_type', '出版社', '一级分类', '二级分类']:
            candidates[col] = candidates[col].astype(str).fillna('missing')

    # 保存ID列和书籍信息（关键修改：只保留不在features中的书籍信息列，避免重复）
    id_cols = ['user_id', 'book_id']
    # 书籍信息列（筛选出不在features中的列，避免与特征重复）
    book_info_cols = [col for col in ['题名', '作者', '出版社', '一级分类', '二级分类'] if col not in features]
    # 候选集最终列：ID列 + 模型特征列 + 非重复的书籍信息列
    candidates = candidates[id_cols + features + book_info_cols]

    print("  - Predicting scores...")
    scores = model.predict_proba(candidates[features])[:, 1]
    candidates['score'] = scores

    # 为每个用户选择最佳推荐
    if len(candidates) == 0:
        print("No valid candidates after feature processing.")
        return

    if candidates['user_id'].isna().any():
        print("Removing candidates with missing user_id...")
        candidates = candidates.dropna(subset=['user_id'])

    if len(candidates) == 0:
        print("No candidates left after removing missing user_id.")
        return

    # 生成原始推荐（每个用户1个）
    recommendations = candidates.loc[candidates.groupby('user_id')['score'].idxmax()]
    output_df = recommendations[['user_id', 'book_id']]
    output_df.to_csv(config.OUTPUT_FILE, index=False)
    print(f"Recommendations saved to {config.OUTPUT_FILE}. Shape: {output_df.shape}")

    # 生成每个用户的top10推荐
    print("  - Generating top10 recommendations per user...")

    # 按用户分组并按分数降序排序，取前10
    top10_recommendations = (
        candidates.sort_values(['user_id', 'score'], ascending=[True, False])
        .groupby('user_id').head(10)  # 取每个用户的前10条
        .reset_index(drop=True)
    )

    # 保留需要的列：用户ID、书籍ID、置信概率、书籍信息（自动适配剩余的书籍信息列）
    top10_output_cols = ['user_id', 'book_id', 'score'] + book_info_cols
    top10_output = top10_recommendations[top10_output_cols].rename(
        columns={'score': '置信概率', '题名': '书名'}  # 重命名列名更直观
    )

    # 保存top10推荐
    top10_output.to_csv(config.TOP10_OUTPUT_FILE, index=False)
    print(f"Top10 recommendations saved to {config.TOP10_OUTPUT_FILE}. Shape: {top10_output.shape}")


# --- Main Execution ---
if __name__ == '__main__':
    cfg = Config()

    # 1. 加载数据
    full_df, model_data_df = load_and_prepare_data(cfg.DATA_FILE)
    # 2. 特征工程
    featured_df = generate_content_embeddings(model_data_df, cfg)
    graph_embeds = generate_graph_embeddings(full_df, cfg)
    featured_df = merge_graph_embeddings(featured_df, graph_embeds, cfg)
    featured_df = generate_statistical_features(featured_df)
    # 3. 训练模型
    model, feature_list = train_catboost_model(featured_df, cfg)
    # 4. 生成推荐
    generate_recommendations(full_df, model, feature_list, cfg)
    print("\nProcess finished successfully.")