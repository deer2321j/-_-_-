import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import warnings
import os
import joblib

# --- Configuration ---
warnings.filterwarnings('ignore')
DATA_FILE = '111data.csv'

# --- Output File Paths ---
TOP_RECOMMENDATIONS_OUTPUT_FILE = '更多数据的f23的final_recommendations.csv'
TOP1_RECOMMENDATIONS_OUTPUT_FILE = '更多数据的f23混推.csv'
# --- Model & Preprocessor Save Paths ---
MODEL_2ND_BORROW_PATH = 'lgbm_model_2nd.joblib'
PREPROCESSOR_2ND_BORROW_PATH = 'preprocessor_2nd.joblib'
MODEL_3RD_BORROW_PATH = 'lgbm_model_3rd.joblib'
PREPROCESSOR_3RD_BORROW_PATH = 'preprocessor_3rd.joblib'
# --- Feature & Model Parameters ---
TEXT_EMBEDDING_DIMS = 128
THIRD_BORROW_PROB_WEIGHT = 1.0  # Weight for 3rd borrow predictions
CONFIDENCE_THRESHOLD = 0.1#0.14  # 人为设定的置信度阈值，低于此值则采用统计推荐
DAYS_THRESHOLD = 8*10^3  # 天数阈值：距离用户当前时间超过此天数的图书将被淘汰

numerical_features = ['总借阅时间(天)', '续借延迟(天)', '续借次数']
categorical_features = ['gender', 'department', 'grade', 'user_type', '是否续借']
text_cols = ['题名', '一级分类', '二级分类', '作者', '出版社']


def prepare_base_data(file_path):
    """加载基础数据并进行初始处理，如计算借阅次数和排名"""
    print("Loading and preparing base data...")
    df = pd.read_csv(file_path)
    df['借阅时间'] = pd.to_datetime(df['借阅时间'], errors='coerce')
    df['还书时间'] = pd.to_datetime(df['还书时间'], errors='coerce')

    # 计算每个用户-图书对的借阅次数和排名
    df['borrow_count'] = df.groupby(['user_id', 'book_id'])['inter_id'].transform('count')
    df['borrow_rank'] = df.groupby(['user_id', 'book_id'])['借阅时间'].rank(method='first').astype(int)

    # 填充关键列中的NaN值
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in categorical_features + text_cols:
        df[col] = df[col].astype(str).fillna('missing')

    return df


def prepare_data_for_2nd_borrow(df):
    """准备第二次借阅预测模型的训练数据"""
    print("\n--- Preparing data for 2nd borrow model ---")
    df_first = df[df['borrow_rank'] == 1].copy()
    df_first['target'] = (df_first['borrow_count'] >= 2).astype(int)

    # 特征工程
    df_first['text_features'] = df_first[text_cols].agg(' '.join, axis=1)

    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df_first[col] = le.fit_transform(df_first[col])
        label_encoders[col] = le

    X_tabular = df_first[numerical_features + categorical_features].values
    y = df_first['target'].values

    print("Generating text embeddings for 2nd borrow model...")
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('svd', TruncatedSVD(n_components=TEXT_EMBEDDING_DIMS, random_state=42))
    ])
    text_embeddings = text_pipeline.fit_transform(df_first['text_features'])

    X_final = np.hstack([X_tabular, text_embeddings])

    preprocessor_payload = {'text_pipeline': text_pipeline, 'label_encoders': label_encoders}
    joblib.dump(preprocessor_payload, PREPROCESSOR_2ND_BORROW_PATH)
    print(f"2nd borrow preprocessor saved to {PREPROCESSOR_2ND_BORROW_PATH}")

    return X_final, y, df_first['user_id']


def prepare_data_for_3rd_borrow(df):
    """准备第三次借阅预测模型的训练数据"""
    print("\n--- Preparing data for 3rd borrow model ---")
    df_ge2 = df[df['borrow_count'] >= 2].copy()

    # 获取每个用户-图书对的第一次和第二次借阅记录
    grouped = df_ge2.groupby(['user_id', 'book_id'])
    first_borrows = grouped.nth(0)
    second_borrows = grouped.nth(1)

    # 以第二次借阅记录为基础构建特征
    data_for_3rd = second_borrows.copy()
    data_for_3rd['target'] = (data_for_3rd['borrow_count'] >= 3).astype(int)

    # 创建序列特征
    time_deltas = second_borrows['借阅时间'].values - first_borrows['还书时间'].values
    time_diff_days = pd.to_timedelta(time_deltas, errors='coerce').days
    data_for_3rd['time_between_borrows_1_2'] = pd.Series(time_diff_days).fillna(999).values

    # 第一次借阅的持续时间
    data_for_3rd['borrow_1_duration'] = first_borrows['总借阅时间(天)'].values

    # 前两次借阅的续借模式(编码为整数)
    data_for_3rd['renewal_pattern_1_2'] = ((first_borrows['是否续借'] == 'True').values.astype(int) * 2 +
                                           (second_borrows['是否续借'] == 'True').values.astype(int))

    # 更新特征列表
    sequential_features = ['time_between_borrows_1_2', 'borrow_1_duration', 'renewal_pattern_1_2']
    all_numerical_features = numerical_features + sequential_features

    # 特征工程
    data_for_3rd['text_features'] = data_for_3rd[text_cols].agg(' '.join, axis=1)

    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        data_for_3rd[col] = le.fit_transform(data_for_3rd[col])
        label_encoders[col] = le

    X_tabular = data_for_3rd[all_numerical_features + categorical_features].values
    y = data_for_3rd['target'].values

    print("Generating text embeddings for 3rd borrow model...")
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('svd', TruncatedSVD(n_components=TEXT_EMBEDDING_DIMS, random_state=42))
    ])
    text_embeddings = text_pipeline.fit_transform(data_for_3rd['text_features'])

    X_final = np.hstack([X_tabular, text_embeddings])

    preprocessor_payload = {'text_pipeline': text_pipeline, 'label_encoders': label_encoders}
    joblib.dump(preprocessor_payload, PREPROCESSOR_3RD_BORROW_PATH)
    print(f"3rd borrow preprocessor saved to {PREPROCESSOR_3RD_BORROW_PATH}")

    return X_final, y, data_for_3rd['user_id']


def train_lightgbm_model(X, y, user_ids_for_split, numerical_feature_count, model_save_path, n_estimators=1000,
                         retrain=False):
    """训练或重新训练LightGBM模型，内部处理特征缩放"""
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
        print(f"Retrained model and scaler saved to {model_save_path}")
        return lgbm, scaler


def generate_recommendations(df, confidence_threshold):
    """生成推荐结果，结合模型预测和历史回退策略"""
    print("\n--- Generating Final Recommendations (Hybrid Strategy) ---")

    # --- 新增：天数阈值过滤逻辑 ---
    print(f"\n--- Applying Days Threshold Filter ({DAYS_THRESHOLD} days) ---")

    # 计算每个用户的当前时间（最后一次有记录的时间）
    print("Calculating current time for each user...")
    user_current_time = df.groupby('user_id')[['借阅时间', '还书时间']].max().max(axis=1)

    # 计算每个用户-图书对的最后一次还书时间
    print("Calculating last return time for each user-book pair...")
    book_last_return_time = df.groupby(['user_id', 'book_id'])['还书时间'].max()

    # 过滤条件：当前时间与最后一次还书时间的天数差 <= 阈值
    valid_books_mask = []
    for (user_id, book_id), last_return_time in book_last_return_time.items():
        current_time = user_current_time[user_id]
        days_diff = (current_time - last_return_time).days
        valid_books_mask.append(days_diff <= DAYS_THRESHOLD)

    # 创建有效的用户-图书对集合
    valid_user_books = set()
    for (user_id, book_id), is_valid in zip(book_last_return_time.index, valid_books_mask):
        if is_valid:
            valid_user_books.add((user_id, book_id))

    print(f"After days threshold filtering: {len(valid_user_books)} valid user-book pairs remain")

    # 加载所有模型和预处理器
    model_2nd_payload = joblib.load(MODEL_2ND_BORROW_PATH)
    model_2nd = model_2nd_payload['model']
    scaler_2nd = model_2nd_payload['scaler']
    preprocessor_2nd = joblib.load(PREPROCESSOR_2ND_BORROW_PATH)
    text_pipeline_2nd = preprocessor_2nd['text_pipeline']
    le_2nd = preprocessor_2nd['label_encoders']

    model_3rd_payload = joblib.load(MODEL_3RD_BORROW_PATH)
    model_3rd = model_3rd_payload['model']
    scaler_3rd = model_3rd_payload['scaler']
    preprocessor_3rd = joblib.load(PREPROCESSOR_3RD_BORROW_PATH)
    text_pipeline_3rd = preprocessor_3rd['text_pipeline']
    le_3rd = preprocessor_3rd['label_encoders']

    all_recommendations = []

    # 第二次借阅预测（应用天数阈值过滤）
    print("\nProcessing all candidates for 2nd borrow with days threshold filter...")
    candidates_2nd = df[df['borrow_count'] == 1].copy()
    if not candidates_2nd.empty:
        # 应用天数阈值过滤
        candidates_2nd['is_valid'] = candidates_2nd.apply(
            lambda row: (row['user_id'], row['book_id']) in valid_user_books, axis=1
        )
        candidates_2nd = candidates_2nd[candidates_2nd['is_valid']]
        print(f"After filtering: {len(candidates_2nd)} candidates remain for 2nd borrow")

        if not candidates_2nd.empty:
            candidates_2nd['text_features'] = candidates_2nd[text_cols].agg(' '.join, axis=1)
            for col in categorical_features:
                known_labels = set(le_2nd[col].classes_)
                candidates_2nd[col] = candidates_2nd[col].apply(lambda x: x if x in known_labels else 'missing')
                if 'missing' not in le_2nd[col].classes_:
                    le_2nd[col].classes_ = np.append(le_2nd[col].classes_, 'missing')
                candidates_2nd[col] = le_2nd[col].transform(candidates_2nd[col])
            X_tab_2nd = candidates_2nd[numerical_features + categorical_features].values
            X_text_2nd = text_pipeline_2nd.transform(candidates_2nd['text_features'])
            X_tab_2nd[:, :len(numerical_features)] = scaler_2nd.transform(X_tab_2nd[:, :len(numerical_features)])
            X_final_2nd = np.hstack([X_tab_2nd, X_text_2nd])
            scores_2nd = model_2nd.predict_proba(X_final_2nd)[:, 1]
            for i, score in enumerate(scores_2nd):
                rec = candidates_2nd.iloc[i]
                all_recommendations.append(
                    {'user_id': rec['user_id'], 'book_id': rec['book_id'], 'score': score, 'model_type': '2nd_borrow'})

    # 第三次借阅预测（应用天数阈值过滤）
    print("\nProcessing all candidates for 3rd borrow with days threshold filter...")
    candidate_pairs_3rd = df[df['borrow_count'] == 2][['user_id', 'book_id']].drop_duplicates()
    if not candidate_pairs_3rd.empty:
        # 应用天数阈值过滤
        candidate_pairs_3rd['is_valid'] = candidate_pairs_3rd.apply(
            lambda row: (row['user_id'], row['book_id']) in valid_user_books, axis=1
        )
        candidate_pairs_3rd = candidate_pairs_3rd[candidate_pairs_3rd['is_valid']]
        print(f"After filtering: {len(candidate_pairs_3rd)} candidates remain for 3rd borrow")

        if not candidate_pairs_3rd.empty:
            df_ge2 = df[df['borrow_count'] >= 2].copy()
            grouped = df_ge2.groupby(['user_id', 'book_id'])
            first_borrows = grouped.nth(0)
            second_borrows = grouped.nth(1)
            candidates_3rd = second_borrows.reset_index().merge(candidate_pairs_3rd, on=['user_id', 'book_id'])
            first_borrows_filtered = first_borrows.reset_index().merge(candidates_3rd[['user_id', 'book_id']],
                                                                       on=['user_id', 'book_id'])
            candidates_3rd = candidates_3rd.sort_values(['user_id', 'book_id']).reset_index(drop=True)
            first_borrows_filtered = first_borrows_filtered.sort_values(['user_id', 'book_id']).reset_index(drop=True)
            time_deltas = pd.to_datetime(candidates_3rd['借阅时间'].values) - pd.to_datetime(
                first_borrows_filtered['还书时间'].values)
            time_diff_days = pd.to_timedelta(time_deltas, errors='coerce').days
            candidates_3rd['time_between_borrows_1_2'] = pd.Series(time_diff_days).fillna(999).values
            candidates_3rd['borrow_1_duration'] = first_borrows_filtered['总借阅时间(天)'].values
            candidates_3rd['renewal_pattern_1_2'] = (
                    (first_borrows_filtered['是否续借'] == 'True').values.astype(int) * 2 + (
                    candidates_3rd['是否续借'] == 'True').values.astype(int))
            sequential_features = ['time_between_borrows_1_2', 'borrow_1_duration', 'renewal_pattern_1_2']
            all_numerical_features_3rd = numerical_features + sequential_features
            candidates_3rd['text_features'] = candidates_3rd[text_cols].agg(' '.join, axis=1)
            for col in categorical_features:
                known_labels = set(le_3rd[col].classes_)
                candidates_3rd[col] = candidates_3rd[col].apply(lambda x: x if x in known_labels else 'missing')
                if 'missing' not in le_3rd[col].classes_:
                    le_3rd[col].classes_ = np.append(le_3rd[col].classes_, 'missing')
                candidates_3rd[col] = le_3rd[col].transform(candidates_3rd[col])
            X_tab_3rd = candidates_3rd[all_numerical_features_3rd + categorical_features].values
            X_text_3rd = text_pipeline_3rd.transform(candidates_3rd['text_features'])
            X_tab_3rd[:, :len(all_numerical_features_3rd)] = scaler_3rd.transform(
                X_tab_3rd[:, :len(all_numerical_features_3rd)])
            X_final_3rd = np.hstack([X_tab_3rd, X_text_3rd])
            scores_3rd = model_3rd.predict_proba(X_final_3rd)[:, 1]
            for i, score in enumerate(scores_3rd):
                rec = candidates_3rd.iloc[i]
                all_recommendations.append(
                    {'user_id': rec['user_id'], 'book_id': rec['book_id'], 'score': score, 'model_type': '3rd_borrow'})

    # --- 模型分数处理与回退逻辑 ---
    print("\n--- Normalizing, Applying Fallback, and Combining Scores ---")
    if not all_recommendations:
        print("No model recommendations were generated.")
        return

    recs_df = pd.DataFrame(all_recommendations)

    # 打印原始候选推荐统计
    print(f"Total raw model recommendations generated: {len(recs_df)}")
    if not recs_df.empty:
        print(f"Breakdown by candidate type:\n{recs_df['model_type'].value_counts()}")

    recs_df['置信概率'] = 0.0
    mask_2nd = recs_df['model_type'] == '2nd_borrow'
    scores_2nd_borrow = recs_df.loc[mask_2nd, 'score']
    mask_3rd = recs_df['model_type'] == '3rd_borrow'
    scores_3rd_borrow = recs_df.loc[mask_3rd, 'score']

    if not scores_2nd_borrow.empty:
        min_2, max_2 = scores_2nd_borrow.min(), scores_2nd_borrow.max()
        recs_df.loc[mask_2nd, '置信概率'] = (scores_2nd_borrow - min_2) / (max_2 - min_2) if max_2 > min_2 else 0.5
    if not scores_3rd_borrow.empty:
        min_3, max_3 = scores_3rd_borrow.min(), scores_3rd_borrow.max()
        recs_df.loc[mask_3rd, '置信概率'] = (scores_3rd_borrow - min_3) / (max_3 - min_3) if max_3 > min_3 else 0.5

    print(f"Applying weight ({THIRD_BORROW_PROB_WEIGHT}x) to 3rd borrow predictions...")
    recs_df.loc[mask_3rd, '置信概率'] *= THIRD_BORROW_PROB_WEIGHT
    recs_df['置信概率'] = recs_df['置信概率'].clip(0, 1)
    recs_df.rename(columns={'model_type': '预测类型'}, inplace=True)

    model_recs_sorted = recs_df.sort_values('置信概率', ascending=False)

    # 确定需要回退的用户
    top1_model_recs = model_recs_sorted.groupby('user_id').head(1)
    low_confidence_users = top1_model_recs[top1_model_recs['置信概率'] < confidence_threshold]['user_id'].unique()
    print(
        f"Found {len(low_confidence_users)} users with top recommendation below threshold {confidence_threshold}. Applying fallback.")

    fallback_recs_list = []
    if len(low_confidence_users) > 0:
        for user_id in low_confidence_users:
            user_history = df[df['user_id'] == user_id]
            if user_history.empty:
                continue

            # 应用天数阈值过滤到回退策略
            user_valid_books = [book_id for (uid, book_id) in valid_user_books if uid == user_id]
            user_history = user_history[user_history['book_id'].isin(user_valid_books)]

            if user_history.empty:
                continue

            max_borrows = user_history['borrow_count'].max()
            top_books = user_history[user_history['borrow_count'] == max_borrows]
            # 按借阅时间倒序排列，选择最近借阅的图书
            top_books_sorted = top_books.sort_values('借阅时间', ascending=False)

            # 选择第一本有效的图书
            fallback_book = None
            for _, book in top_books_sorted.iterrows():
                if (user_id, book['book_id']) in valid_user_books:
                    fallback_book = book
                    break

            if fallback_book is not None:
                fallback_recs_list.append({
                    'user_id': user_id,
                    'book_id': fallback_book['book_id'],
                    'score': max_borrows,  # 使用借阅次数作为分数
                    '置信概率': 1.0,  # 回退策略的置信度设为最高
                    '预测类型': 'historical_fallback'
                })

    # 组合最终推荐
    high_confidence_recs = model_recs_sorted[~model_recs_sorted['user_id'].isin(low_confidence_users)]
    fallback_df = pd.DataFrame(fallback_recs_list)
    final_recs_df = pd.concat([high_confidence_recs, fallback_df], ignore_index=True)
    final_recs_sorted = final_recs_df.sort_values('置信概率', ascending=False)

    # --- 打印最终推荐来源统计 ---
    top1_recs_final = final_recs_sorted.groupby('user_id').head(1)
    final_user_counts = top1_recs_final.groupby('预测类型')['user_id'].nunique()
    model_2nd_users = final_user_counts.get('2nd_borrow', 0)
    model_3rd_users = final_user_counts.get('3rd_borrow', 0)
    fallback_users = final_user_counts.get('historical_fallback', 0)
    total_final_users = top1_recs_final['user_id'].nunique()

    print("\n--- Final Recommendation Statistics ---")
    print(f"Total unique users in final recommendations: {total_final_users}")
    print(f"Users from '2nd_borrow' model: {model_2nd_users}")
    print(f"Users from '3rd_borrow' model: {model_3rd_users}")
    print(f"Users from 'historical_fallback': {fallback_users}")

    # --- 输出最终结果文件 ---
    output_columns = ['user_id', 'book_id', 'score', '置信概率', '预测类型']

    # 为每个用户获取前10名推荐
    top10_recs = final_recs_sorted.groupby('user_id').head(10)
    top10_recs[output_columns].to_csv(TOP_RECOMMENDATIONS_OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nTop 10 final recommendations saved to {TOP_RECOMMENDATIONS_OUTPUT_FILE}")

    # 为每个用户获取前1名推荐用于提交
    top1_recs = final_recs_sorted.groupby('user_id').head(1)
    top1_recs[output_columns].to_csv(TOP1_RECOMMENDATIONS_OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"Top 1 final recommendations for submission saved to {TOP1_RECOMMENDATIONS_OUTPUT_FILE}")


if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at '{DATA_FILE}'")
    else:
        base_df = prepare_base_data(DATA_FILE)

        # 第二次借阅模型
        X_2nd, y_2nd, users_2nd = prepare_data_for_2nd_borrow(base_df)
        best_iter_2nd, _ = train_lightgbm_model(X_2nd, y_2nd, users_2nd, len(numerical_features), MODEL_2ND_BORROW_PATH)
        train_lightgbm_model(X_2nd, y_2nd, users_2nd, len(numerical_features), MODEL_2ND_BORROW_PATH,
                             n_estimators=max(1, best_iter_2nd), retrain=True)

        # 第三次借阅模型
        X_3rd, y_3rd, users_3rd = prepare_data_for_3rd_borrow(base_df)
        num_features_3rd = len(numerical_features) + 3  # 3个序列特征
        best_iter_3rd, _ = train_lightgbm_model(X_3rd, y_3rd, users_3rd, num_features_3rd, MODEL_3RD_BORROW_PATH)
        train_lightgbm_model(X_3rd, y_3rd, users_3rd, num_features_3rd, MODEL_3RD_BORROW_PATH,
                             n_estimators=max(1, best_iter_3rd), retrain=True)

        # 生成组合推荐
        generate_recommendations(base_df, confidence_threshold=CONFIDENCE_THRESHOLD)

        print("\nProcess completed successfully.")