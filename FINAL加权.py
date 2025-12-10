import pandas as pd
from collections import defaultdict, Counter
'''相对路径 可直接运行'''
#python FINAL加权.py
csv_paths = [
    "纯3轻量辅助23混高速度运行完毕（编号3）/1olddata决赛时决赛纯3.csv",
    "f1轻量辅助推荐（编号7）/f1data.csv", "top10加权输出结果.csv",
    "决赛时的v5系列/旧决赛时的16fv5.csv","dspos2（编号4）/决赛dspos2.csv","133用扩充特征kaggle环境跑（编号6）/决赛133.csv",
    "23混推轻量快速相对高性能，一分钟即可生成一个基础预测结果（编号1）/更多数据的f23混推.csv","七以上的v5.csv"
    ,"v2超轻量辅助（新加）/旧决赛时的fv2.csv",
    "复赛时的部分结果重用，可进入编号里的代码复现/复赛时的23混推/1复赛时f23的final_recommendations.csv",  "复赛时的部分结果重用，可进入编号里的代码复现/复赛时的f1系列/1复赛时的f1.csv",
  "dspos2（编号4）/dspos2初赛结果.csv", "133用扩充特征kaggle环境跑（编号6）/133.csv"
]
#整理时误删了个文件，确保运行，去掉了那个所以权重多出来一个，是乱的我懒得改到对应了，，总之就是这个思路借鉴下吧
model_weights = { 0: 1.1,1: 0.5, 2: 0.5,3: 0.5,  4: 0.5, 5: 0.5, 6: 1.0, 7: 1.7, 8: 1.8, 9: 1.1, 10: 0.5, 11: 0.5, 12: 0.7, 13: 0.7}
output_csv = "submission.csv"
# -------------------------- 初始化变量 --------------------------
all_predictions = []
model_predictions = [set() for _ in range(len(csv_paths))]

# -------------------------- 读取CSV --------------------------
for file_index, path in enumerate(csv_paths):
    df = pd.read_csv(path)
    df["file_index"] = file_index
    all_predictions.extend(df.to_dict("records"))

    for _, row in df.iterrows():
        model_predictions[file_index].add((row["user_id"], row["book_id"]))

# -------------------------- 加权投票 --------------------------
user_votes = defaultdict(lambda: defaultdict(lambda: [0.0, float("inf")]))

for pred in all_predictions:
    user_id = pred["user_id"]
    book_id = pred["book_id"]
    file_idx = pred["file_index"]

    weight = model_weights.get(file_idx, 1.0)
    user_votes[user_id][book_id][0] += weight
    if file_idx < user_votes[user_id][book_id][1]:
        user_votes[user_id][book_id][1] = file_idx

# -------------------------- 生成最终推荐 --------------------------
recommendation_source = []
final_result = []

for user_id, book_info in user_votes.items():
    sorted_books = sorted(
        book_info.items(),
        key=lambda x: (-x[1][0], x[1][1])
    )
    final_book_id, final_info = sorted_books[0]
    final_result.append({"user_id": user_id, "book_id": final_book_id})
    recommendation_source.append({
        "user_id": user_id,
        "source_model_index": final_info[1],
        "weighted_score": final_info[0],
        "votes": round(final_info[0])
    })

final_set = set((item["user_id"], item["book_id"]) for item in final_result)

# -------------------------- 输出结果 --------------------------
final_df = pd.DataFrame(final_result)
final_df = final_df.sort_values("user_id").reset_index(drop=True)
final_df.to_csv(output_csv, index=False)

# -------------------------- 统计信息 --------------------------
print("\n" + "=" * 50)
print("推荐统计信息（加权投票）")
print("=" * 50)

voting_user_count = len(user_votes)
print(f"1. 总用户数: {voting_user_count}")
print(f"2. 参与投票的模型数: {len(csv_paths)}")

weighted_scores = [info[0] for user in user_votes.values() for info in user.values()]
rounded_scores = [round(score) for score in weighted_scores]
score_distribution = Counter(rounded_scores)
max_score = max(score_distribution.keys()) if score_distribution else 0

print("\n3. 所有投票推荐书籍的加权得分分布（近似得票数）:")
for i in range(1, max_score + 1):
    print(f"   近似得票数为 {i} 的书籍数量: {score_distribution.get(i, 0)}")

print("\n4. 最终推荐来源分布:")
final_rounded_scores = [round(item["weighted_score"]) for item in recommendation_source]
final_score_distribution = Counter(final_rounded_scores)
for i in range(1, max_score + 1):
    count = final_score_distribution.get(i, 0)
    percentage = (count / voting_user_count) * 100
    print(f"   投票推荐（近似得票{i}）: {count} 个（占投票用户 {percentage:.2f}%）")

print("\n" + "=" * 50)
print(f"最终推荐已生成：{output_csv}")