# 手机文档扫描与 OCR 课程项目

`docscan` 将手机拍摄的纸张和书页矫正、去阴影并导出为白底 PDF。它支持书脊拆页、同页多拍合成和中文可搜索 PDF。OCR 对照实验使用合成数据，实拍讲义用于展示扫描效果。

## 快速生成扫描 PDF

在项目根目录用 PowerShell 运行以下命令。自带样例是合成文档，导出的图像 PDF 位于 `outputs/quick_scan/book.pdf`。

```powershell
$env:PYTHONUTF8 = '1'
python -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -e .
& $py -m docscan to-pdf --input examples/synthetic/sample.jpg --source synthetic --page-format a4 --output outputs/quick_scan
```

## OCR 与实验环境

实测组合为 Windows 11 x64、CPython 3.13.7、PaddlePaddle 3.3.0、PaddleOCR 3.3.1、PaddleX 3.3.13。锁文件对应 Windows x64 Python 3.13。使用前面建好的虚拟环境安装完整依赖。

```powershell
& $py -m pip install -r requirements.win-py313.lock
& $py -m pip install --no-build-isolation --no-deps -e .
& $py -m pip check
& $py -m docscan --help
& $py -m docscan doctor
```

其他平台可在隔离环境尝试 `python -m pip install -e ".[ocr,report,test]"` 并运行 `doctor` 与 `ocr-check` 核对实际支持。依赖与模型来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 从可分发样例开始

以下样例来自程序渲染。`--corners` 接受已知四角 JSON，省略时走自动定位。`--pdf-mode image` 生成普通图像 PDF。

```powershell
& $py -m docscan validate-data --manifest examples/synthetic/manifest.csv
& $py -m docscan scan --input examples/synthetic/sample.jpg --corners examples/synthetic/sample.corners.json --method E --ocr none --pdf-mode image --output outputs/quick_image
```

首次使用 OCR 时下载两份官方 PP-OCRv5 权重并校验归档哈希。离线运行时，`doctor` 与模型加载都核对缓存文件的 SHA-256。可通过 YAML 的 `ocr.model_dir` 指向另一处已准备的权重目录。CPU 推理关闭了文档方向、文档展平、文字行方向和 MKL-DNN。模型识别错误、空图、缺权重和缓存损坏分别保留状态。

```powershell
& $py -m docscan prepare-models
& $py -m docscan scan --input examples/synthetic/sample.jpg --corners examples/synthetic/sample.corners.json --method E --ocr paddle --pdf-mode searchable --output outputs/quick_ocr
& $py -m docscan verify-pdf --input outputs/quick_ocr/scan.pdf --expected-text outputs/quick_ocr/text.txt --ocr-json outputs/quick_ocr/ocr.json --background outputs/quick_ocr/enhanced_gray.png --keyword '校园阅读'
```

`--output` 不存在时直接使用该目录。目录已存在时会创建新的 `run_id` 子目录，后续命令用 CLI 打印的 `run_dir`。`run.json` 记录源图哈希、实际配置、设备、模型、时间与输出。`ocr_raw.json` 保留 PaddleOCR 的坐标和字段，`ocr.json` 使用调用图像的像素坐标，`text.txt` 按单栏阅读顺序输出。

## 纸张与书页转白底 PDF

用 `to-pdf` 将单张照片或目录合成 `book.pdf`。单页、书籍封面和 A4 纸用 `--layout single`（默认）。一本摊开的书用 `--layout spread`，每张照片分为左页和右页。目录按文件名数字排序，例如 `2.jpg` 在 `10.jpg` 前。程序按纸面的局部亮度去阴影，再轻量锐化笔画并保留字内空隙的灰阶。识别出的彩色和灰度插图区域保留原有色调。

斜拍时，程序可从完整矩形纸页的两条正交方向估计宽高比。完整页会清理贴着边缘的折痕与细长残影，并保护识别到的印刷边线。裁切页优先保留拍到的边缘内容。宽高比估计参考 [Zhang 与 He 的拍照校正方法](https://www.microsoft.com/en-us/research/publication/whiteboard-scanning-image-enhancement/)。

默认按 300 DPI 生成并嵌入页图，A4 纵向为 2480 × 3508 像素。`--dpi` 可设为 72 至 400，`--dpi 150` 适合更小的文件。DPI 调整的是页图像素，PDF 的物理尺寸由原页规格决定。`run.json` 在每页记录嵌入图尺寸与实测有效 DPI。

```powershell
& $py -m docscan to-pdf --input '照片.jpg' --page-format a4 --output outputs/paper_pdf
& $py -m docscan to-pdf --input '书页照片目录' --layout spread --output outputs/book_pdf
```

同一张完整纸页拍了 2 至 4 次时，可以把这些照片放在一个目录，用 `--same-page` 配准并合成一页。程序检查匹配点覆盖、全页墨迹及局部字形。`run.json` 保留光流微调前后的差异区域和局部平移比对结果。疑似内容冲突时返回 `capture_mismatch`，分开保存已处理的页图，供 `review_pages` 复查。该模式使用 `--layout single` 和 `--dewarp none`。

```powershell
& $py -m docscan to-pdf --input '同一页照片目录' --same-page --output outputs/merged_page
```

当角点被遮住或缺失时，程序根据可见长边推算四角，再分割实际可见的纸面。没有足够边线时，使用能分出的可见区域。纸外和缺角区域填白，A4 选项补足页面比例并采用 210 × 297 mm 的实际 PDF 页面尺寸。原图只剩窄条时，可加 `--page-orientation portrait` 或 `landscape` 指定纸张方向。输入已经拍不到的内容对应留白，页内印刷文字、图表与插图保留为图像。浅色印刷和反光区域需要看输出页复查。

如果书脊判断不稳，整张作为一页保留，页码列在 `review_pages`。可以传 `--gutter 0.49`，或用 `--gutter '0.46,0.51'` 表示书脊顶部和底部在 EXIF 转正照片中的横向比例。单页边界难以区分时可传 `--corners`。每页图像和纸面遮罩保存在 `pages/`，`run.json` 记录边界、分割和留白情况。纸面定位不可靠时保守保留可见内容并标记复查。

```powershell
& $py -m docscan to-pdf --input '摊开的书.jpg' --layout spread --pdf-mode searchable --output outputs/book_search
& $py -m docscan verify-pdf --input outputs/book_search/book.pdf --expected-text outputs/book_search/book.txt
```

`--pdf-mode searchable` 为每个处理后的页面加可搜索文字层，使用前先准备 OCR 权重。严重弯曲的单页可选 UVDoc，建议对照 `before_dewarp.png` 检查展平效果。两种默认图像流程使用现有图像处理依赖即可。

双栏排版可加 `--reading-order two-column`。程序按每页中线分栏，在跨栏标题之间先读左栏再读右栏。默认 `rows` 按行读取，适合单栏。两个选项也可用于 `scan` 和 `ocr-check`。传给 `verify-pdf --ocr-json` 的文件应包含图像宽高、识别行文本和四边形坐标。

```powershell
& $py -m docscan prepare-models --dewarp
& $py -m docscan to-pdf --input '摊开的书.jpg' --layout spread --dewarp uvdoc --output outputs/book_flattened
```

## 批处理与合成复现

批量扫描按文件名排序。`--corners` 对目录输入接收同名 `*.corners.json` 的目录。`--auto-first` 在无合适自动候选时使用标注，运行记录会单列人工补救。目录扫描会继续处理坏图之后的照片，任一图片失败时返回非零。

```powershell
& $py -m docscan make-samples --output data/synthetic --seed 42
& $py -m docscan validate-data --manifest data/synthetic/manifest.csv
& $py -m docscan scan --input data/synthetic --corners data/synthetic --auto-first --ocr none --output outputs/batch_images
& $py -m docscan benchmark --manifest data/synthetic/manifest.csv --split test --source synthetic --mode shared-corners --config configs/experiment.yaml --output outputs/reproduce_test
& $py -m docscan benchmark --manifest data/synthetic/manifest.csv --split test --source synthetic --mode auto --config configs/experiment.yaml --output outputs/reproduce_auto
& $py -m docscan summarize --run-dir outputs/reproduce_test
& $py -m docscan build-report --run-dir outputs/reproduce_test --auto-run-dir outputs/reproduce_auto --output reports
```

`make-samples` 需要新目录或空目录。可用 `--font` 或 `DOCSCAN_FONT` 指定支持中英文的 TTF、OTF 或 TTC。相同依赖、字体文件和种子可复核样例哈希。`benchmark` 输出逐图 `results.csv`、按方法与拍摄条件汇总的 `summary.csv` 及参数快照。`analysis/` 存放 CER 或覆盖率图、耗时表、四条件光照对照、全页与局部案例。测试配置来自开发集，含配置及模型哈希。

## 真实实验

按 [拍摄与数据导入](docs/data.md) 准备 `data/real/manifest.csv`。正式对照先核对 4 份开发文档和 6 份测试文档，每份四种拍摄条件各一张。以下命令使用默认配置作为真实开发起点。可对照 `configs/dev_wide.yaml`，选定后把对应开发运行目录传给 `freeze-config`。

```powershell
& $py -m docscan validate-data --manifest data/real/manifest.csv --require-real-plan
& $py -m docscan benchmark --manifest data/real/manifest.csv --split dev --source real --config configs/default.yaml --output outputs/real_dev
& $py -m docscan freeze-config --dev-run outputs/real_dev --output configs/experiment.real.yaml
& $py -m docscan benchmark --manifest data/real/manifest.csv --split test --source real --config configs/experiment.real.yaml --output outputs/real_benchmark
& $py -m docscan benchmark --manifest data/real/manifest.csv --split test --source real --mode auto --config configs/experiment.real.yaml --output outputs/real_auto
& $py -m docscan summarize --run-dir outputs/real_benchmark
& $py -m docscan build-report --run-dir outputs/real_benchmark --auto-run-dir outputs/real_auto --output reports
```

CER 对预测和真值采用 NFKC、去空白后做字符编辑对齐，保留标点和英文大小写。汇总采用累计错误数除以累计真值长度。数据、模型和预热协议见 [实验说明](docs/experiment.md)。

## 测试与常见错误

```powershell
& $py -m pytest -m 'not ocr_integration'
& $py -m pytest -m ocr_integration
```

GitHub 的 [Windows validation](.github/workflows/windows.yml) 在推送和拉取请求中安装核心依赖、运行回归测试并核对合成 A4 PDF。在 Actions 手动运行时启用 `ocr`，会增加独立的 OCR 集成任务，校验模型文件并确认集成用例实际执行。

| 错误或状态 | 处理 |
| --- | --- |
| `detection_failed` | 用角点叠加图或输入图复核边界，再传 `--corners`。A 组原图 OCR 可独立使用。 |
| `invalid_corners` | 检查坐标系、EXIF 归一化尺寸及左上、右上、右下、左下顺序。 |
| `ocr_model_missing` | 运行 `prepare-models`，或将两份已准备的权重放入配置目录。 |
| `model_file_changed` | 根据报错定位缓存文件。移开相应模型目录，再运行 `prepare-models` 或 `prepare-models --dewarp`。 |
| `invalid_ocr_json` | 使用对应扫描的 `ocr.json`，复核文件中的图像宽高、识别行与坐标。 |
| 首次 Paddle 导入遇到缓存权限 | Paddle 3.3.0 首次导入会初始化用户缓存目录。用可写用户缓存的本地环境初始化后，可继续离线执行。 |
| venv 或 pytest 的临时目录权限错误 | 将当前 PowerShell 进程的 `TEMP` 和 `TMP` 指向项目内可写目录，再建 venv 或运行测试。 |
| `no_searchable_text` | OCR 返回空结果。`ocr.json` 保留 `no_text`，如需查看图像版请用 `--pdf-mode image`。 |
| `pdf_font_missing_glyph` | 用 `--pdf-font` 或配置 `pdf.font` 指定覆盖这些字符的字体。 |
| `config_not_frozen` | 在同一来源的开发集完成 A–F 配对，运行 `freeze-config` 后再测。 |
| OCR 集成测试 `skipped` | 此环境的真实推理未验证。按 `doctor` 输出补装依赖和模型。 |

## 项目目录

- `src/docscan/` 包含命令行入口、几何矫正、图像增强、OCR 和 PDF 导出。
- `examples/synthetic/` 提供可直接运行的合成样例。完整合成数据集由 `make-samples` 生成。
- `configs/` 保存默认参数和合成实验的冻结配置。
- `docs/` 说明拍摄数据格式和对照实验协议。
- `tests/` 包含几何、书页、OCR 和 PDF 的回归检查。

## 原有仓库备注

11
