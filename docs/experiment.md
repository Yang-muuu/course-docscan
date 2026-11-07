# 实验协议与合成选参

## 配对实验

同一图像分别进入 A–F。A 直接读取 EXIF 归一化的彩色照片，B 使用透视矫正灰度，C 和 D 在 B 上做 Otsu 或自适应阈值，E 使用光照校正灰度，F 用和 D 相同的自适应阈值参数处理 E。B–F 共享标注四角和矫正图。A/B 同时包含几何与灰度变化，E/B 和 F/D 对应两条光照校正比较。

用固定的两份 PP-OCRv5 模型、CPU、置信度阈值和缩放。每次运行初始化一次模型，在所选集合第一张成功识别文字的照片上完成预热。前处理时间包含 EXIF 解码、对应分支处理和推理前缩放。对于 B–F，共享矫正与灰度耗时在各分支完整计入，方便比较单独使用某一方法时的分项。OCR 时间只覆盖 `predict`。模型准备、加载、预热、写盘和 CER 对齐另计。该协议固定方法执行顺序，时间仅用于本机描述性比较。

`ok` 与 `no_text` 纳入 CER。`error` 保留结果行和原因，CER 留空。各方法的有效率和六组完整配对数一并输出。不完整运行生成覆盖率图，报告暂不列整体优胜。`summary.csv` 的微平均按各方法有效记录的字符总数加权。

## 独立自动定位评价

自动候选按标注四边形计算 IoU 和最大角点距离。当前门槛为 IoU 至少 0.92、最大角点误差不超过图像对角线的 3%。具备文字区域标注时，还需保留全部标注区域的四个顶点。没有文字区域时记录 `geometry_only` 或失败原因。采用标注角点重试成功的 `manual_rescue` 单列。`auto_results.csv` 和叠加图可用于复查。

## 合成配置的冻结依据

开发集 2 份文档、8 张图比较了短边比例 0.045 和 0.075，其余配置固定。按 E 和 F 的合计编辑次数除以合计真值字符数选参，平局选择较小的核。默认核为 13/3376，宽核为 12/3376，因此测试用 `configs/experiment.yaml` 冻结 0.075。可在 `outputs/verification/dev_selection.json` 查到开发运行 ID、计数、数据与模型哈希。

```powershell
& $py -m docscan benchmark --manifest data/synthetic/manifest.csv --source synthetic --split dev --config configs/default.yaml --output outputs/recheck_dev_default
& $py -m docscan benchmark --manifest data/synthetic/manifest.csv --source synthetic --split dev --config configs/dev_wide.yaml --output outputs/recheck_dev_wide
Import-Csv outputs/recheck_dev_default/summary.csv | Where-Object { $_.condition -eq 'ALL' -and $_.method -in 'E','F' }
Import-Csv outputs/recheck_dev_wide/summary.csv | Where-Object { $_.condition -eq 'ALL' -and $_.method -in 'E','F' }
```

合成页的文字和四角由程序生成，印刷体有可见合成标记。模拟渐变、投影和阴影的样本用于验证处理协议与识别产物。真实照片补齐后单独验证自动定位，按真实开发集选参再评价真实测试集。
