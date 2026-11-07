from __future__ import annotations

import os
from pathlib import Path

from .common import DocscanError


def chinese_font(explicit: Path | None = None, *, prefer_static: bool = False) -> Path:
    if explicit is None and os.getenv("DOCSCAN_FONT"):
        explicit = Path(os.environ["DOCSCAN_FONT"])
    if explicit is not None:
        if explicit.is_file():
            return explicit
        raise DocscanError("font_missing", f"Chinese font does not exist: {explicit}")

    directories = [Path.home() / ".local/share/fonts", Path("/usr/share/fonts/opentype/noto"),
                   Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental")]
    if os.getenv("WINDIR"):
        directories.insert(0, Path(os.environ["WINDIR"]) / "Fonts")
    names = ("NotoSansSC-VF.ttf", "NotoSansSC-Regular.ttf", "NotoSansCJK-Regular.ttc",
             "msyh.ttc", "simhei.ttf", "PingFang.ttc", "STHeiti Medium.ttc")
    if prefer_static:
        names = (*names[1:], names[0])
    for directory in directories:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise DocscanError("font_missing", "Set DOCSCAN_FONT or pass --font to a CJK TTF/OTF/TTC file")


def sample_font(path: Path, size: int):
    from PIL import ImageFont

    font = ImageFont.truetype(str(path), size)
    try:
        if b"Regular" in font.get_variation_names():
            font.set_variation_by_name("Regular")
    except OSError:
        pass  # Static font, already at its designed weight.
    return font

