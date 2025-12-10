# Global AI Algorithm Challenge — Multi‑Model Library Recommendation

A production‑ready, multi‑model ensemble that generates book‑borrowing recommendations. Each sub‑model trains and predicts independently, and results are fused with a two‑layer weighted voting strategy to produce the final submission file.

## Highlights
- Multi‑model pipeline with independent training/inference per folder
- Two‑layer weighted voting to improve robustness and accuracy
- Full reproduction path with per‑model instructions in subfolders
## Notes

- Final competition submission model files and intermediate CSV artifacts are preserved offline. Because of size limits, large artifacts such as `.joblib`, `.pkl`, and `.csv` are not hosted in this repository. For a complete, fully reproducible package, follow the per‑model READMEs step by step, or contact via email: `a1992423911@dlmu.edu.cn` to request a full compressed archive.
- 中文说明：最终决赛提交的模型文件、过程 CSV 产物均已保存；`joblib`、`pkl`、`csv` 等文件无法上传到此仓库。如需完整复现，请按步骤在各目录运行，或联系邮箱：`a1992423911@dlmu.edu.cn` 获取完整压缩包。
## Quick Start

From the project root, run:

```bash
python FINAL加权.py
```
- Output: `submission.csv`
- All scripts use relative paths; run them from the project root.

## Data & Inputs

- Semi‑final data file: `1data.csv`
- Final data file: `111data.csv`
- Many sub‑model folders include their own data copies. Follow each folder’s README for exact placement.

## Full Reproduction

Generate per‑model outputs, then perform staged fusion.

```bash
# 1) Per‑model outputs (run inside each folder)
#    23混推 / v5 / dspos2 / 133 / f1 / v2 / 决赛classic_autoML
#    Follow the README inside each folder to produce its CSV output.

# 2) Back to project root: fuse v5 results
python 整合rv5到最终投票.py    # => 七以上的v5.csv

# 3) Fuse top‑10 weighted outputs
python Top10加权融合.py         # => top10加权输出结果.csv

# 4) Final two‑layer weighted voting
python FINAL加权.py              # => submission.csv
```

## Repository Layout (key items)

- `FINAL加权.py` — Final two‑layer weighted voting to produce `submission.csv`
- `Top10加权融合.py` — Top‑10 weighted fusion to produce `top10加权输出结果.csv`
- `整合rv5到最终投票.py` — v5 results integration to produce `七以上的v5.csv`
- Sub‑model folders (examples): `23混推/`, `v5/`, `dspos2/`, `133/`, `f1/`, `v2/`, `决赛classic_autoML/`
- Docs & helpers: `整体README帮助理解.txt`, `环境依赖.txt`

## Requirements

- Python 3.8+ recommended
- Install dependencies as documented in each sub‑model folder
- File encoding: UTF‑8; ensure column names match the scripts



## How It Works (high level)

1. Train/Infer each sub‑model independently to produce its ranking or score CSV.
2. Integrate intermediate results (e.g., v5 integration) for broader coverage.
3. Apply top‑10 weighted fusion to stabilize ranking across models.
4. Run the final two‑layer weighted voting, yielding `submission.csv` for competition submission.

## Troubleshooting

- Always run scripts from the project root so relative paths resolve correctly.
- If a script cannot find data, verify `1data.csv` / `111data.csv` are placed as instructed in the folder README.
- If encoding issues occur, convert inputs to UTF‑8 and check column names.

## License

Choose and add a license (e.g., MIT or Apache‑2.0) to clarify usage and contributions.
