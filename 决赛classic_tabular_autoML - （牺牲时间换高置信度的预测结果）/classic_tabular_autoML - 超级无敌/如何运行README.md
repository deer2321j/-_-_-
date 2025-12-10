# 完整运行流程说明
## **重要提示：关于计算资源**
本项目是一个计算密集型任务，特别是 **Step2 (文本编码)** 和 **Step6 (AutoGluon训练)** 阶段，对内存和CPU/GPU有较高要求。
- **文本编码 (Step2)**：涉及大规模的BERT嵌入计算，耗时较长且消耗大量内存。
- **模型训练 (Step6)**：AutoGluon会探索多种模型并进行集成，需要强大的计算能力以在合理时间内完成。
我们团队在开发和最终运行时，租用了云服务器完成全流程。在个人电脑运行可能因内存不足或性能问题失败。请在运行前评估您的硬件环境，建议服务器上运行（建议32GB内存以上），训练完成后直接运行：python step7_autogluon_predict.py 可生成预测结果，我们训练好的文件都已经保存在对应的文件夹中，如果您没有破坏项目中已保存的中间文件 ，直接在终端输入命令 python step7_autogluon_predict.py 运行即可。
---
本文档说明如何在本项目中从原始数据到模型预测，完整、可重复地执行全流程。
## **快速预测教程 **
确保您的 `Models/` 文件夹包含完整的预训练模型，您可以跳过计算密集型的 **Step6 (训练)**，直接进行预测。请按以下顺序执行脚本，以生成预测所需的候选集并运行预测：

### 最快速运行方案：
如果您没有破坏项目中已保存的中间文件 ，请直接运行：python step7_autogluon_predict.py 可生成预测结果

### 较为完整的运行方案 （只跳过计算密集的step6)

```powershell
# 1. 生成所有必需的特征
python step1_data_cleaning.py
python step2_bert_encoding.py
python step3_merge_features.py
python step4_target_construction_bert_full.py
python step3.9_generate_recent_books_data.py #可选
python step4.1_merge_recent_books.py #可选
python step4.2_add_recent_book_pca.py #可选 

# 2. 清理预测候选集特征
python step5_clean_features.py

# 3. 运行预测
python step7_autogluon_predict.py
```

执行完毕后，最终的提交文件将生成在 `results/submission.csv`。

---

## 1. 环境与依赖

- 操作系统：Windows（PowerShell）
- Python：建议 3.9+（3.8 及以上均可）
- 依赖包：见 `requirements.txt`

示例安装（PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. 数据与目录结构

- 原始数据目录：`raw_data/`
  - `inter_final_选手可见.csv`（核心借阅记录）
  - `user.csv`（用户信息）
  - `books_with_features.csv`（图书特征）
  - `douban_books.csv`（豆瓣评分）
- 处理后数据目录：`processed_data/`
  - 各 Step 产物均写入此目录

请确保 `raw_data/inter_final_选手可见.csv` 存在且包含至少：`user_id`, `book_id`, `借阅时间`（时间列用于排序与切分）。

---

## 3. 流程总览

推荐的标准运行顺序如下（括号内为关键输入输出）：

1) **Step1 清洗与特征工程**（输入：`raw_data/...` → 输出：`processed_data/step1_features.csv`）
2) **Step2 文本向量编码**（输入：图书名/摘要 → 输出：`processed_data/step2_*` 诸多嵌入与 PCA 文件）
3) **Step3 特征合并**（输入：Step1 + Step2 → 输出：`processed_data/step3_final_features.csv`）
4) **Step4 目标与候选构造**（输入：`inter_final_选手可见.csv` + Step3 → 输出：`step4_train_data.csv`、`step4_targets.csv`、`step4_prediction_candidates.csv`）
5) **Step3.9 生成最近借阅书籍数据**（输入：`inter_final_选手可见.csv` → 输出：`step3.9_train_with_recent_books.csv`、`step3.9_prediction_candidates_with_recent_books.csv`）
6) **Step4.1 合并最近借阅书籍特征**（输入：Step4 + Step3.9 → 输出：覆盖写回 `step4_train_data.csv`、`step4_prediction_candidates.csv`，增加 `recent_book_*` 列）
7) **Step4.2 为最近书与候选书添加 PCA 向量**（输入：Step4.1 输出 → 输出：为 `step4_train_data.csv`/`step4_prediction_candidates.csv` 增加 `recent_book_*_pca_*` 与 `book_name_pca_*` 列）
8) **Step5 特征清理**（输入：Step4.2 → 输出：`step5_clean_train.csv`、`step5_clean_candidates.csv`）
9) **Step6 训练（AutoGluon）**（输入：Step5 → 输出：模型/评估）
10) **Step7 预测（AutoGluon）**（输入：模型 + 候选 → 输出：预测结果）
11) **提交产物合并**（`results/` 下脚本）

---

## 4. 分步执行与命令示例

以下命令均在项目根目录执行：

- **Step1：数据清洗与特征工程**

```powershell
python step1_data_cleaning.py
```

- **Step2：文本（书名/摘要）编码与 PCA**

```powershell
python step2_bert_encoding.py
```

- **Step3：合并结构化特征与文本嵌入**

```powershell
python step3_merge_features.py
```

- **Step4：目标与候选构造**

```powershell
python step4_target_construction_bert_full.py
```

- **Step3.9：生成最近借阅书籍数据（可选）**

```powershell
python step3.9_generate_recent_books_data.py
```

- **Step4.1：合并最近借阅书籍特征（可选）**

```powershell
python step4.1_merge_recent_books.py
```

- **Step4.2：为最近书与候选书添加 PCA 向量（可选）**

```powershell
python step4.2_add_recent_book_pca.py
```

- **Step5：特征清理与筛选**

```powershell
python step5_clean_features.py
```

- **Step6：训练**

```powershell
python step6_autogluon_full.py
```

- **Step7：预测**

```powershell
python step7_autogluon_predict.py
```

- **提交产物生成**

```powershell
python data_generate/create_submission_template.py
```

---

## 5. 验证清单（关键断点）

- **Step1**：`processed_data/step1_features.csv` 行列数合理。
- **Step3**：`processed_data/step3_final_features.csv` 存在且按 `book_id` 合并成功。
- **Step4**：`processed_data/step4_prediction_candidates.csv` 行数合理；每位用户约 100 条候选。
- **Step4.1**：`step4_train_data.csv` 与 `step4_prediction_candidates.csv` 覆盖写回后存在 `recent_book_*` 列。
- **Step4.2**：`step4_train_data.csv` 与 `step4_prediction_candidates.csv` 覆盖写回后存在 `recent_book_*_pca_*` 列。
- **Step5**：`step5_clean_*` 文件存在，特征数符合预期。

---

## 6. 快速开始（可行链路）

跑结构化特征 → 候选生成 → 训练/预测：

```powershell
python step1_data_cleaning.py
python step3_merge_features.py
python step4_target_construction_bert_full.py
python step5_clean_features.py
python step6_autogluon_train.py
python step7_autogluon_predict.py
```
