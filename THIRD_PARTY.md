# 第三方依赖与来源

应用代码在此项目中编写。实现参考下列官方 API 与文档，模型直接使用公开推理权重。

| 部分 | 来源和许可 |
| --- | --- |
| PaddleOCR、PaddlePaddle、PaddleX | Apache 2.0。实测版本分别为 3.3.1、3.3.0、3.3.13。[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) · [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) · [PaddleX](https://github.com/PaddlePaddle/PaddleX) |
| PP-OCRv5 mobile 检测和识别权重 | 作者模型卡标注 Apache 2.0。`prepare-models` 从 PaddleOCR 3.3.1 文档链接的官方推理仓下载，项目记录下载的归档和文件 SHA-256。[下载来源](https://www.paddleocr.ai/v3.3.1/en/version3.x/pipeline_usage/OCR.html) · [检测模型卡](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det) · [识别模型卡](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) |
| UVDoc（可选） | 书页展平使用 PaddleOCR 3.3.1 提供的 UVDoc 推理权重。`prepare-models --dewarp` 独立下载并核对 SHA-256。实拍书页烟测使用该模块文档公开的演示照片。[UVDoc 模块、权重和演示图](https://www.paddleocr.ai/v3.3.1/en/version3.x/module_usage/text_image_unwarping.html) |
| OpenCV | 使用 `opencv-contrib-python` 4.10.0.84。OpenCV 为 Apache 2.0，Python wheel 构建仓为 MIT。[OpenCV](https://github.com/opencv/opencv/blob/4.10.0/LICENSE) · [wheel 构建说明](https://github.com/opencv/opencv-python#licensing) |
| PyMuPDF 1.26.7 | AGPL 或商业许可。`TextWriter` 写入不可见的文字层，默认使用 PyMuPDF 随包的 CJK 字体 Droid Sans Fallback Regular。[PyMuPDF 许可](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright) · [CJK 字体与度量](https://pymupdf.readthedocs.io/en/latest/font.html) |
| 合成字形和图表 | 合成图实测使用系统上的 Noto Sans SC，SIL Open Font License 1.1。图表和 Word 实测使用 Windows 的 Microsoft YaHei。样例归档包含渲染像素、字号和字体标识，不附字体文件。[Noto CJK 字体许可](https://github.com/notofonts/noto-cjk/blob/main/Sans/LICENSE) |
| 其他运行依赖 | NumPy（BSD 3-Clause）、Pillow（HPND）、PyYAML（MIT）、Matplotlib（Matplotlib License）、python-docx（MIT）、pytest（MIT）。完整版本见 `requirements.win-py313.lock`，各发行包附带各自的许可文本。 |
| Windows 桌面运行库 | 随包 CPython 与 Tcl/Tk 来自构建机的官方 Windows Python 3.13 安装。CPython 许可位于 `runtime/LICENSE.txt`，Tk 许可位于 `runtime/tcl/tk8.6/license.terms`。各 wheel 的许可位于 `runtime/Lib/site-packages` 下的发行包元数据目录。[CPython 3.13.7 源码](https://www.python.org/downloads/release/python-3137/) · [PyMuPDF 1.26.7 源码](https://pypi.org/project/PyMuPDF/1.26.7/#files) |
| Microsoft Visual C++ 运行库 | `runtime` 中附带构建机已安装的 x64 Visual C++ 可再发行库，供 PyMuPDF 和 PaddlePaddle 加载。[Microsoft 可再发行文件说明](https://learn.microsoft.com/cpp/windows/determining-which-dlls-to-redistribute) |

代码依赖 OpenCV 的几何变换、闭运算和阈值接口，OCR 使用 PaddleOCR 3.x 的 `predict`，PDF 验收包括 PyMuPDF 的坐标抽取和 Poppler 页面渲染。[OpenCV 几何](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html) · [OpenCV 形态学](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) · [OpenCV 阈值](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) · [PyMuPDF TextWriter](https://pymupdf.readthedocs.io/en/latest/textwriter.html)
