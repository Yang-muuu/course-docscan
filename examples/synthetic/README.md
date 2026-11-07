# 一页合成扫描样例

从种子 42 的 `syn-01` 文档选择斜拍阴影样例。`sample.jpg` 含可见合成标记，`sample.corners.json` 保存投影角点与文字区域，`syn-01.txt` 是生成文字。先用 `validate-data --manifest examples/synthetic/manifest.csv` 校验，再按项目 README 执行单张图像或 OCR 流程。用于功能复现，完整开发与测试集由 `make-samples` 生成。
