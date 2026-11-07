# 拍摄与数据导入

用 10 份内容不同的平整 A4 文档，保持正向、单栏、中英文印刷体。每份拍 4 张，目标 40 张。预先指定 4 份为开发集，余下 6 份为测试集。同一内容的全部照片使用同一个 `doc_id`、`split` 与转录文件。没有收齐时如实填写实际条数。

| `condition` | 拍摄要求 |
| --- | --- |
| `front_even` | 相机尽量平行于纸面，均匀光照 |
| `tilt_even` | 斜拍，保持整张纸和四角清楚可见 |
| `front_shadow` | 正拍，用侧光或遮挡部分光源，形成局部阴影 |
| `tilt_shadow` | 斜拍并保留局部阴影 |

使用同一设备和分辨率，避免闪光、反光、运动模糊和实体遮住文字。保留原始 JPG/PNG，在 `notes` 记录有明显差异的光源、背景、曝光或字号。真实文档尽量选择无个人信息的自编印刷材料。

## 文件与字段

1. 把照片、转录与角点文件放入 `data/real/` 下的目录。把 `manifest.example.csv` 复制为 `manifest.csv` 并填入每张实际照片的一行。路径均相对该 CSV。
2. `image_id` 全局唯一，`doc_id` 对应文档内容。两者使用字母、数字、连字符、点或下划线，首字符为字母或数字。`image_id` 须能作为 Windows 目录名，末尾用字母、数字、连字符或下划线，并避开 `CON`、`NUL`、`COM1` 等保留名称。`split` 取 `dev` 或 `test`，`source` 取 `real`，`page_orientation` 取 `portrait` 或 `landscape`。`notes` 可以为空。
3. UTF-8 转录按单栏从上到下输入，保留中英文大小写、数字和标点。优先从实际打印源抄录，并对照纸面复核。实验前固定转录。OCR 预测独立输出，评价阶段才读取转录。
4. 对 EXIF 归一化后的照片标四角，点顺序为左上、右上、右下、左下。`image_size` 用 `[宽, 高]`，坐标以左上角像素为 `(0, 0)`。如标注工具使用未旋转的原始像素，声明 `raw_image_pixels` 并填写原图尺寸，程序按 EXIF 转换。选填 `text_polygons`（每个正文区域四个顶点，同一坐标系），供自动定位评价检查文字是否被裁掉。

CSV 行示意。替换文件路径和条件，确保对应实际拍摄的照片。

```csv
doc01-front_even,doc01,photos/doc01-front_even.jpg,transcripts/doc01.txt,corners/doc01-front_even.corners.json,dev,front_even,real,portrait,窗边均光
```

角点文件字段示意，坐标需替换为实际标注。

```json
{
  "coordinate_space": "exif_normalized_pixels",
  "image_size": [1200, 1600],
  "points": [[120, 100], [1080, 120], [1100, 1490], [100, 1480]],
  "order": "tl_tr_br_bl",
  "source": "real"
}
```

```powershell
python -m docscan validate-data --manifest data/real/manifest.csv --require-real-plan
```

收集过程中的部分数据可用普通 `validate-data` 检查。正式实验使用 `--require-real-plan`，同时校验真实来源、开发与测试文档数量，以及每份文档的四种条件各一张。

`make-samples` 生成的是程序渲染的合成集。每页含合成标记，清单、数据集元数据及角点文件都记录 `synthetic`。角点和文字区域来自生成变换。字体与依赖相同、种子相同时可复核文件哈希。
