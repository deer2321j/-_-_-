# 23混推（编号1）使用说明

## 快速开始
- 确保当前目录是项目根目录后，进入本目录：
  - `cd 23混推轻量快速相对高性能，一分钟即可生成一个基础预测结果（编号1）`
- 运行脚本（训练+预测，约一分钟内完成）：
  - `python BookBorrowRec-LightGBM（轻量一分钟可运行完毕23混推）.py`
- 输出文件位于本目录：
  - `f23混推.csv`（每用户Top1，用于提交/融合）
  - `final_recommendations.csv`（每用户Top10，用于融合）

## 两种运行模式
- 直接运行（默认）：
  - 脚本会读取 `111data.csv` 并训练两套轻量模型后生成结果。
  - 数据文件路径常量：`BookBorrowRec-LightGBM（轻量一分钟可运行完毕23混推）.py:16`。
- 加载已上传模型（略过重新训练）：
  - 将已上传的模型与预处理文件重命名为脚本默认名，然后直接运行同上命令：
    - `更多数据的f23lgbm_model_2nd.joblib` → `lgbm_model_2nd.joblib`（`BookBorrowRec-LightGBM（轻量一分钟可运行完毕23混推）.py:23`）
    - `更多数据的f23preprocessor_2nd.joblib` → `preprocessor_2nd.joblib`（`BookBorrowRec-LightGBM（轻量一分钟可运行完毕23混推）.py:24`）
    - `更多数据的f23lgbm_model_3rd.joblib` → `lgbm_model_3rd.joblib`（`BookBorrowRec-LightGBM（轻量一分钟可运行完毕23混推）.py:25`）
    - `更多数据的f23preprocessor_3rd.joblib` → `preprocessor_3rd.joblib`（`BookBorrowRec-LightGBM（轻量一分钟可运行完毕23混推）.py:26`）
  - 由于模型轻量，直接训练与加载已上传模型的时间差距不大，两种方式均可快速得到结果。

## 依赖与环境
- Python 3.8+
- 依赖安装（根目录或本目录均可）：
  - `pip install pandas numpy lightgbm scikit-learn joblib`

## 输出与融合衔接
- 本目录输出的 `f23混推.csv`、`final_recommendations.csv` 会被根目录的融合脚本读取：
  - `Top10加权融合.py`（生成 `top10加权输出结果.csv`）
  - `FINAL加权.py`（生成最终 `submission-噜啦啦.csv`）
- 输出文件名定义：`BookBorrowRec-LightGBM（轻量一分钟可运行完毕23混推）.py:20-22`，写出逻辑：`BookBorrowRec-LightGBM（轻量一分钟可运行完毕23混推）.py:411-417`。

