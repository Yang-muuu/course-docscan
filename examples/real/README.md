# 实拍去影示例

三张照片来自用户提供的 `datadata.zip`，取自对应 PDF 的嵌入原图。

| 文件 | 来源 | 拍摄条件 |
| --- | --- | --- |
| `01-light-shadow.jpg` | PDF 1 第 3 页 | 散页，浅阴影和光照不均 |
| `02-dark-shadow.jpg` | PDF 4 第 4 页 | 散页，深阴影横跨正文和页脚 |
| `03-book-shadow.jpg` | PDF 8 第 3 页 | 横拍书页，书脊弯曲和深阴影 |

完整解压后双击 `DocScan.exe`。点击「打开实拍示例」后，原图直接显示纸页边界，拖动四角可微调。点击「开始处理」后，在结果区选择 A、B、C。保存 PDF 时使用每张照片各自选定的版本。A 均衡正文和白底，B 优先压低阴影边缘及浅底纹，C 保留公式细线和图形灰阶。

在解压后的 `DocScan` 目录运行以下命令，可将三张候选分别保存为 PNG。

```powershell
& .\runtime\python.exe -X utf8 -B -m docscan candidates --input .\examples\real --output .\示例结果
```
