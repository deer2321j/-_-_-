import pandas as pd
from collections import defaultdict, Counter

# -------------------------- 1. 配置参数 --------------------------
# 输入CSV文件路径列表
csv_paths = ["决赛时的v5系列/旧决赛时的32fv5.csv",
             "决赛时的v5系列/旧决赛时的64fv5.csv",
             "决赛时的v5系列/旧决赛时的16fv5.csv",
             "决赛时的v5系列/旧决赛时的256fv5.csv",
             "决赛时的v5系列/旧决赛时的128fv5.csv",
             "决赛时的v5系列/64fv5.csv","决赛时的v5系列/32fv5.csv","决赛时的v5系列/128fv5.csv","决赛时的v5系列/16fv5.csv","决赛时的v5系列/512fv5.csv"]
#靠前的为用决赛数据的预测结果文件，后面的为复赛时的预测结果文件，共同投票，可帮助筛选出加了数据后预测结果仍不变的高置信度高稳定性用户
# 最终输出的CSV路径
output_csv = "七以上的v5.csv"
# 最低投票数阈值：只保留得票数大于等于该值的推荐
min_votes = 6# 可根据需要调整此参数
# -------------------------- 2. 读取所有CSV并记录文件优先级 --------------------------
all_predictions = []
# 存储每个模型的推荐结果，用于后续交集计算
model_predictions = [set() for _ in range(len(csv_paths))]  # 每个元素是一个集合，存储(user_id, book_id)元组

for file_index, path in enumerate(csv_paths):
    df = pd.read_csv(path)
    df["file_index"] = file_index
    all_predictions.extend(df.to_dict("records"))

    # 记录当前模型的所有推荐
    for _, row in df.iterrows():
        model_predictions[file_index].add((row["user_id"], row["book_id"]))

# -------------------------- 3. 按user_id聚合投票 --------------------------
user_votes = defaultdict(lambda: defaultdict(lambda: [0, float("inf")]))

for pred in all_predictions:
    user_id = pred["user_id"]
    book_id = pred["book_id"]
    file_idx = pred["file_index"]

    user_votes[user_id][book_id][0] += 1  # 累加票数
    # 记录最低的文件索引（优先级）
    if file_idx < user_votes[user_id][book_id][1]:
        user_votes[user_id][book_id][1] = file_idx

# -------------------------- 4. 按规则选最终推荐（添加最低票数过滤） --------------------------
final_result = []
recommendation_source = []

for user_id, book_info in user_votes.items():
    # 先按票数降序、优先级升序排序
    sorted_books = sorted(
        book_info.items(),
        key=lambda x: (-x[1][0], x[1][1])
    )
    # 过滤出得票数大于等于最低阈值的书籍
    filtered_books = [book for book in sorted_books if book[1][0] >= min_votes]

    # 只有存在符合条件的书籍时才添加到最终结果
    if filtered_books:
        final_book_id, final_info = filtered_books[0]
        final_result.append({"user_id": user_id, "book_id": final_book_id})
        recommendation_source.append({
            "user_id": user_id,
            "source_model_index": final_info[1],
            "votes": final_info[0]
        })

# 将最终推荐结果转为集合，方便计算交集
final_set = set((item["user_id"], item["book_id"]) for item in final_result)

# -------------------------- 5. 生成最终CSV --------------------------
final_df = pd.DataFrame(final_result)
final_df = final_df.sort_values("user_id").reset_index(drop=True)
final_df.to_csv(output_csv, index=False)

# -------------------------- 6. 统计并打印有价值的信息 --------------------------
print("\n" + "=" * 50)
print("推荐统计信息")
print("=" * 50)

# 6.1 基本统计
total_original_users = len(user_votes)  # 原始总用户数
total_final_users = len(final_result)  # 过滤后保留的用户数
print(f"1. 原始总用户数: {total_original_users}")
print(f"2. 过滤后保留的用户数（得票数≥{min_votes}）: {total_final_users}")
print(f"3. 参与投票的模型数: {len(csv_paths)}")
print(f"4. 最低投票数阈值: {min_votes}")

# 6.2 所有推荐书籍的得票分布统计
vote_counts = [info[0] for user in user_votes.values() for info in user.values()]
vote_distribution = Counter(vote_counts)
max_vote = max(vote_distribution.keys()) if vote_distribution else 0

print("\n5. 所有推荐书籍的得票分布:")
for i in range(1, max_vote + 1):
    print(f"   得票数为 {i} 的书籍数量: {vote_distribution.get(i, 0)}")

# 6.3 最终推荐的得票分布
final_vote_counts = [item["votes"] for item in recommendation_source]
final_vote_distribution = Counter(final_vote_counts)

print("\n6. 最终推荐书籍的得票分布:")
for i in range(min_votes, max_vote + 1):  # 从阈值开始统计
    count = final_vote_distribution.get(i, 0)
    percentage = (count / total_final_users) * 100 if total_final_users > 0 else 0
    print(f"   得票数为 {i} 的推荐占比: {count} 个 ({percentage:.2f}%)")

# 6.4 各模型的推荐被选中的数量
source_distribution = Counter(item["source_model_index"] for item in recommendation_source)
total_selected = sum(source_distribution.values())

print("\n7. 各模型推荐被最终选中的数量:")
for i in range(len(csv_paths)):
    count = source_distribution.get(i, 0)
    percentage = (count / total_selected) * 100 if total_selected > 0 else 0
    print(f"   模型 {i + 1} ({csv_paths[i]}) 被选中: {count} 次 ({percentage:.2f}%)")

# 6.5 各模型与最终推荐结果的交集统计
print("\n8. 各模型与最终推荐结果的交集统计:")
for i in range(len(csv_paths)):
    # 计算当前模型与最终结果的交集数量
    intersection_count = len(model_predictions[i] & final_set)
    # 该模型的总推荐数
    total_recommendations = len(model_predictions[i])

    # 计算比例
    percentage = (intersection_count / total_recommendations) * 100 if total_recommendations > 0 else 0
    print(
        f"   模型 {i + 1} ({csv_paths[i]}) 与最终结果的交集: {intersection_count} 个，占该模型总推荐的 {percentage:.2f}%")

# 6.6 结果文件信息
print("\n" + "=" * 50)
print(f"最终推荐已生成：{output_csv}")
print(f"注：仅保留了得票数≥{min_votes}的推荐结果")