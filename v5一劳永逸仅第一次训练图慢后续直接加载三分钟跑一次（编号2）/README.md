# v5一劳永逸（编号2）使用说明
仅首次从头训练慢，后续直接加载缓存三分钟跑一次即可。
完整复现时可加载我们的缓存与模型，即可在数分钟内生成推荐结果。
## 设计思路
- 一次较完整的特征工程与模型训练（较慢），产出可复用的缓存与模型；后续仅加载缓存与模型，即可在数分钟内生成推荐结果。
- 特征组合：文本（TF-IDF+SVD 256维）+ 图结构（Node2Vec 64维）+ 统计特征；模型采用 CatBoost 分类器。

## 目录与关键文件
- 脚本：`1复赛赛本地top10.py`（训练+生成推荐）
- 数据：`111data.csv`
- 预训练与缓存（示例，可能为不同维度版本）：
  - `旧决赛16fcatboost_model_v5.joblib`（16维版本模型）
  - `旧决赛时16ftext_pipeline_v5.joblib`（文本管线）
  - `旧决赛时fgraph_embeddings_v5.pkl`（图嵌入缓存）
- 输出：
  - `旧决赛时的16fv5.csv`（每用户Top1）
  - `旧决赛时ftop10_recommendations_v5.csv`（每用户Top10）

脚本默认配置（256维版本）位于：`v5一劳永逸…/1复赛赛本地top10.py:24-41`
- `DATA_FILE`：`111data.csv`（`1复赛赛本地top10.py:25`）
- `OUTPUT_FILE`：`256fv5.csv`（`1复赛赛本地top10.py:26`）
- `TOP10_OUTPUT_FILE`：`ftop10_recommendations_v5.csv`（`1复赛赛本地top10.py:27`）
- 缓存路径：文本管线、图嵌入、模型、特征清单（`1复赛赛本地top10.py:29-32`）
- 文本维度 256（`1复赛赛本地top10.py:35`），图嵌入维度 64（`1复赛赛本地top10.py:36`）

## 快速开始
- 在本目录运行：
  - 训练+预测（首次较慢，后续更快）：`python 1复赛赛本地top10.py`
- 生成的输出：
  - `256fv5.csv`（每用户Top1）
  - `ftop10_recommendations_v5.csv`（每用户Top10）
- 首次运行会生成/加载缓存：
  - 文本管线缓存（`1复赛赛本地top10.py:85-106`）
  - 图结构嵌入缓存（`1复赛赛本地top10.py:109-129`）

## 两种使用模式
- 直接训练（建议首次运行）：
  - 读取 `111data.csv`，执行特征工程与 CatBoost 训练，再生成推荐。
  - 训练流程与最佳迭代选择：`1复赛赛本地top10.py:179-259`（含验证集早停与全量重训）。
- 加载已上传模型与缓存（快速）：
  - 将现有的缓存与模型文件重命名为脚本默认名，或修改 `Config` 中的路径以匹配现有文件：
    - 文本管线：将 `旧决赛时16ftext_pipeline_v5.joblib` 改为 `256ftext_pipeline_v5.joblib`
    - 图嵌入：保持为 `fgraph_embeddings_v5.pkl`
    - 模型：将 `旧决赛16fcatboost_model_v5.joblib` 改为 `fcatboost_model_v5.joblib`
    - 特征清单：若有 16维版本，请改为 `fcatboost_features_v5.pkl`
  - 运行脚本将直接加载缓存与模型，数分钟内生成推荐。

## 输出与融合衔接
- Top10文件用于融合：`ftop10_recommendations_v5.csv`（`1复赛赛本地top10.py:325-342`）
- Top1文件用于最终投票：`256fv5.csv`（`1复赛赛本地top10.py:318-323`）
- 根目录融合脚本：
  - `Top10加权融合.py` 需要引用 v5 的 Top10 文件；如默认路径不匹配，请将其改为 `v5一劳永逸…/旧决赛时ftop10_recommendations_v5.csv` 或本脚本生成的 `ftop10_recommendations_v5.csv`。
  - `FINAL加权.py` 默认读取 `v5一劳永逸…/旧决赛时的16fv5.csv`；如使用 256维版本，请将 `csv_paths` 中对应条目调整为本目录生成的 `256fv5.csv`（`FINAL加权.py:8` 与 `readme帮助理解.txt` 的说明）。

## 依赖与环境
- Python 3.8+
- 安装依赖：
  - `pip install pandas numpy scikit-learn catboost joblib tqdm networkx node2vec`
- 注意：`Node2Vec` 训练中 `workers=32`（`1复赛赛本地top10.py:121-125`），在本地环境请根据 CPU 核心数调整；首次训练较慢，后续加载缓存即可快速运行。
