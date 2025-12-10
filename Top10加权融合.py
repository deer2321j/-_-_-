import pandas as pd

# 配置文件路径与对应权重（权重和可以不为1）
csv_weights = {
    "v2超轻量辅助（新加）/旧决赛时的v2top10_recommendations.csv": 0.5,
    "决赛时的v5系列/旧决赛时ftop16_recommendations_v5.csv": 0.3,
    "决赛时的v5系列/旧决赛时ftop256_recommendations_v5.csv": 0.2,
}

# 输出文件路径
output_file = "top10加权输出结果.csv"

# ---------------------- 权重归一化处理 ----------------------
total_original_weight = sum(csv_weights.values())
normalized_weights = {
    file: weight / total_original_weight
    for file, weight in csv_weights.items()
}
print(f"原始权重总和: {total_original_weight}，已自动归一化为总和1的权重")
print("归一化后权重:", normalized_weights)
# ------------------------------------------------------------

# 存储所有推荐数据的列表
all_recommendations = []

# 读取并处理每个推荐文件（增加内部标准化步骤）
print("\n开始读取并处理推荐文件...")
for file_path, weight in normalized_weights.items():
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查必要列是否存在
        required_columns = ['user_id', 'book_id', '置信概率', '书名', '作者']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"文件 {file_path} 缺少必要列: {', '.join(missing_columns)}")

        # ---------------------- 新增：文件内置信概率标准化 ----------------------
        # 使用min-max标准化将置信概率缩放到0-1范围
        prob_col = df['置信概率']
        min_prob = prob_col.min()
        max_prob = prob_col.max()

        # 处理特殊情况：如果所有值都相同（避免除以0）
        if max_prob == min_prob:
            df['标准化置信概率'] = 0.5  # 统一设为0.5（中间值）
            print(f"文件 {file_path} 的置信概率全部相同，标准化后均为0.5")
        else:
            # 标准化公式：(x - min) / (max - min)
            df['标准化置信概率'] = (prob_col - min_prob) / (max_prob - min_prob)
            print(f"文件 {file_path} 置信概率标准化完成（原始范围: {min_prob:.4f}~{max_prob:.4f}）")
        # ----------------------------------------------------------------------

        # 添加归一化权重列并保留必要字段
        processed_df = df[['user_id', 'book_id', '标准化置信概率', '书名', '作者']].copy()
        processed_df['normalized_weight'] = weight  # 模型权重

        all_recommendations.append(processed_df)
        print(f"成功处理 {file_path} (归一化权重: {weight:.4f})，包含 {len(df)} 条记录")

    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")

# 检查是否有有效数据
if not all_recommendations:
    print("没有有效数据可供处理，程序退出")
    exit()

# 合并所有推荐数据
combined_df = pd.concat(all_recommendations, ignore_index=True)
print(f"\n所有文件合并完成，共 {len(combined_df)} 条记录")


# 按用户和书籍分组，计算加权置信度（使用标准化后的置信概率）
def calculate_weighted_score(group):
    """基于标准化置信概率和归一化权重计算综合得分"""
    # 加权和 = 标准化置信概率 × 模型权重 的总和
    weighted_sum = (group['标准化置信概率'] * group['normalized_weight']).sum()
    return pd.Series({
        'weighted_confidence': weighted_sum,  # 综合得分（因权重已归一化，总和为1）
        '书名': group['书名'].iloc[0],
        '作者': group['作者'].iloc[0]
    })


# 分组计算
grouped = combined_df.groupby(['user_id', 'book_id']).apply(calculate_weighted_score).reset_index()
print(f"分组计算完成，去重后得到 {len(grouped)} 条(user_id, book_id)组合")

# 按用户分组，取加权置信度最高的1本
final_recommendations = grouped.sort_values(
    by=['user_id', 'weighted_confidence'],
    ascending=[True, False]
).groupby('user_id').head(1)

# 调整输出列顺序
final_recommendations = final_recommendations[
    ['user_id', 'book_id', '书名', '作者', 'weighted_confidence']
]

# 保存结果
final_recommendations.to_csv(output_file, index=False, encoding='utf-8')

# 输出统计信息
print(f"\n最终推荐结果已保存至 {output_file}")
print(f"共为 {final_recommendations['user_id'].nunique()} 个用户生成推荐")
print(f"推荐结果总条数: {len(final_recommendations)}")
