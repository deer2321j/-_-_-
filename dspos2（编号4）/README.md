# dspos2（编号4）使用说明（Kaggle环境）
同编号133用扩充特征（编号6），建议阅读编号133，两个同样环境几乎同样逻辑，只是输入不同用的特征不同，运行好一个后直接运行第二个，避免反复搭环境）
运行过程出现的警告信息对最终结果无影响，直接忽略即可
## 快速开始（我们自己的环境下很容易包冲突，但在kaggle默认环境上只需pip torch_geometric一个包即可运行，因此强烈推荐在 Kaggle 上运行）
- 打开 Kaggle，创建一个 Notebook，启用免费 GPU（默认即可）。
- 在第一个单元执行：
```bash
pip install torch_geometric
```
- 将本目录作为 Kaggle Dataset 导入（包含 `111data.csv`、模型文件如 `final_model.pth` 
  - 加载已上传模型快速预测：直接加载模型文件，用 `111data.csv` 生成预测输出。
  - 重新训练再预测：在 Kaggle 免费 GPU 上约半小时即可完成全流程复现。
- 预测输出文件（本目录中）：
  - `决赛dspos2.csv`（决赛数据输出）
  - `dspos2初赛结果.csv`（初赛数据输出）
  - `top10_recommendations.csv`（每用户Top10，可用于融合）
## 为什么推荐 Kaggle
- 默认环境仅需安装一个包：`torch_geometric`，无需复杂依赖配置，便利性高。
- 提供免费 GPU，计算更快、更稳定；我们在 Kaggle 上验证用时约半小时即可完成复现。
- 为保证与我们提交结果一致，建议在 Kaggle 上按上述步骤重新运行。

## 运行步骤建议

  - 在 Notebook 中：
    - 读取 `111data.csv`
    - 加载目录中的模型文件（如 `final_model.pth` ）
    - 执行推理

## 注意
- 目录与文件名称请保持不变，以确保融合脚本可直接找到对应文件。
- 若在本地运行，请至少安装 `torch_geometric` 与 PyTorch 对应版本；但仍推荐使用 Kaggle 以保证环境与结果一致。