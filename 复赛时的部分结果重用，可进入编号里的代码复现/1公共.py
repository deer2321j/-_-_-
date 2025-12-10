import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv("16submission.csv")
df2 = pd.read_csv("最终提交的submission.csv")


# 获取前两列的列名
key_columns = df1.columns[:2]  # 取前两列作为判断依据

# 检查两个DataFrame的前两列是否一致（列名）
if not list(key_columns) == list(df2.columns[:2]):
    print("警告：两个文件的前两列列名不匹配，可能影响匹配结果")

# 找出前两列相同的公共键（去重）
common_keys = pd.merge(
    df1[key_columns].drop_duplicates(),
    df2[key_columns].drop_duplicates(),
    how='inner',
    on=list(key_columns)
)

# 获取公共部分的数量（不同的键组合数量）
common_count = len(common_keys)
print(f"前两列相同的公共键数量为: {common_count}")

# 从df1中筛选出前两列属于公共键的所有行
df1_common = df1.merge(common_keys, on=list(key_columns), how='inner')

# 从df2中筛选出前两列属于公共键的所有行
df2_common = df2.merge(common_keys, on=list(key_columns), how='inner')

# 合并两个文件的公共部分
result = pd.concat([df1_common, df2_common], ignore_index=True)

# 保存结果
csv_path = '测试过程.csv'
result.to_csv(csv_path, index=False)
print(f"已保存前两列相同的公共部分到 {csv_path}")