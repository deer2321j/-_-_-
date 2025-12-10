# 133用扩充特征（编号6）使用说明（Kaggle环境）

## 快速开始（我们自己的环境下很容易包冲突，但在kaggle默认环境上只需pip torch_geometric一个包即可运行，因此强烈推荐在 Kaggle 上运行）
运行好这个后直接运行编号四的两个同样环境几乎同样逻辑，只是输入不同用的特征不同，，避免反复搭环境
运行过程出现的警告信息对最终结果无影响，直接忽略即可
- 打开 Kaggle，创建一个 Notebook，启用免费 GPU（默认即可）。
- 在第一个单元执行：
pip install torch_geometric
- 将本目录作为 Kaggle Dataset 导入（包含 `111data.csv`、`final_model.pth`）。
- 两种运行方式任选其一，二者耗时差距不大（模型轻量）：
  - 加载已上传模型快速预测：直接加载 `final_model.pth`，用 `111data.csv` 生成预测输出。
  - 重新训练再预测：在 Kaggle 免费 GPU 上约半小时即可完成全流程复现。
- 预测输出文件：
  - `决赛133.csv`（输入111data决赛数据）
  - `133.csv`（输入1data初赛数据）
## 为什么推荐 Kaggle
- 默认环境仅需安装一个包：`torch_geometric`，无需复杂依赖配置，便利性高。
- 提供免费 GPU，计算更快、更稳定；我们在 Kaggle 上验证用时约半小时即可完成复现。
- 为保证与我们提交结果一致，建议在 Kaggle 上按上述步骤重新运行。

## 运行步骤建议
- 模式A：加载已上传模型（最快）
  - 在 Notebook 中：
    - 读取 `111data.csv`
    - 加载 `final_model.pth`
    - 执行推理，生成 `决赛133.csv` 与 `133.csv`
- 模式B：重新训练（高一致性）
  - 在 Notebook 中：
    - 读取 `111data.csv`
    - 执行扩充特征构造与训练流程（使用默认参数）
    - 保存训练产物并进行推理，生成 `决赛133.csv` 与 `133.csv`

## 与根目录融合的衔接
- 根目录的融合脚本会读取本目录输出：
  - `FINAL加权.py` 使用 `133用扩充特征kaggle环境跑（编号6）/决赛133.csv` 与 `133用扩充特征kaggle环境跑（编号6）/133.csv`。
- 在根目录依次运行：
  - `python 整合rv5到最终投票.py`
  - `python Top10加权融合.py`
  - `python FINAL加权.py`（生成最终 `submission-噜啦啦.csv`）

## 注意
- 目录与文件名称请保持不变，以确保融合脚本可直接找到对应文件。
- 若在本地运行，请至少安装 `torch_geometric` 与 PyTorch 对应版本；但仍推荐使用 Kaggle 以保证环境与结果一致。