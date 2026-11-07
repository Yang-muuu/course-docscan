from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.shared import Inches, Pt, RGBColor

EAST_ASIA = "Microsoft YaHei"


def _element(parent, name: str, **attributes):
    node = OxmlElement(f"w:{name}")
    for key, value in attributes.items():
        node.set(qn(f"w:{key}"), str(value))
    parent.append(node)
    return node


def _style(document, name: str, size: float, color: str = "000000", bold: bool = False):
    style = document.styles[name]
    style.font.name = "Arial"
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor.from_string(color)
    style.font.bold = bold
    style.font.italic = False
    rpr = style.element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:eastAsia"), EAST_ASIA)
    fmt = style.paragraph_format
    ppr = style.element.get_or_add_pPr()
    border = ppr.find(qn("w:pBdr"))
    if border is not None:
        ppr.remove(border)
    return fmt


def _table(document, headings: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headings))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    if len(headings) == 2:
        widths = (0.85, 5.62)
    elif len(headings) == 7:
        widths = (0.66, 1.12, 0.86, 0.86, 0.86, 0.96, 1.15)
    else:
        widths = (0.64, 0.65, 1.33, 1.22, 1.22, 1.41)
    for column, width in zip(table.columns, widths, strict=True):
        column.width = Inches(width)
    properties = table._tbl.tblPr
    borders = _element(properties, "tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        _element(borders, edge, val="single", sz=4, color="D9D9D9")
    margins = _element(properties, "tblCellMar")
    for edge in ("top", "bottom", "left", "right"):
        _element(margins, edge, w=68 if edge in {"top", "bottom"} else 105, type="dxa")
    _element(table.rows[0]._tr.get_or_add_trPr(), "tblHeader", val="1")
    for row_index, values in enumerate([headings, *rows]):
        row = table.rows[0] if row_index == 0 else table.add_row()
        _element(row._tr.get_or_add_trPr(), "cantSplit")
        for index, (cell, value, width) in enumerate(zip(row.cells, values, widths, strict=True)):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.12
            p.paragraph_format.keep_with_next = True
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if len(headings) == 2 and index == 1 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(value)
            run.font.size = Pt(9.5)
            if row_index == 0:
                run.bold = True
                run.font.color.rgb = RGBColor.from_string("FFFFFF")
                _element(cell._tc.get_or_add_tcPr(), "shd", val="clear", fill="304456")
            elif row_index % 2 == 0:
                _element(cell._tc.get_or_add_tcPr(), "shd", val="clear", fill="F3F6F8")


def _link(paragraph, label: str, url: str) -> None:
    target = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    link = _element(paragraph._p, "hyperlink")
    link.set(qn("r:id"), target)
    run = _element(link, "r")
    fmt = _element(run, "rPr")
    _element(fmt, "color", val="255778")
    _element(fmt, "u", val="single")
    _element(run, "t").text = label


def write_docx(path: Path, title: str, subtitle: str, blocks: list, figures: Path) -> None:
    document = Document()
    document.core_properties.title = title
    document.core_properties.subject = subtitle
    document.core_properties.author = ""
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.left_margin = section.right_margin = Inches(0.88)
    section.top_margin, section.bottom_margin = Inches(0.79), Inches(0.76)
    section.footer_distance = Inches(0.42)
    normal = _style(document, "Normal", 10.5)
    normal.space_after, normal.line_spacing = Pt(5), 1.20
    title_format = _style(document, "Title", 22, bold=True)
    title_format.space_before, title_format.space_after = Pt(0), Pt(6)
    title_format.keep_with_next = True
    subtitle_format = _style(document, "Subtitle", 12)
    subtitle_format.space_before, subtitle_format.space_after = Pt(0), Pt(13)
    subtitle_format.keep_with_next = True
    for level, size in ((1, 15), (2, 12)):
        heading = _style(document, f"Heading {level}", size, bold=True)
        heading.space_before, heading.space_after = Pt(11 if level == 1 else 9), Pt(4)
        heading.keep_with_next = True
    caption = _style(document, "Caption", 9.5, color="344150")
    caption.space_before, caption.space_after = Pt(5), Pt(9)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _element(footer._p, "fldSimple", instr="PAGE")
    document.add_paragraph(title, "Title")
    document.add_paragraph(subtitle, "Subtitle")
    for kind, value in blocks:
        if kind == "p":
            document.add_paragraph(value)
        elif kind in {"h1", "h2"}:
            document.add_heading(value, 1 if kind == "h1" else 2)
        elif kind == "table":
            _table(document, *value)
        elif kind == "caption":
            document.add_paragraph(value, "Caption")
        elif kind == "image":
            filename, label = value
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            shape = paragraph.add_run().add_picture(str(figures / filename), width=Inches(6.37))
            shape._inline.docPr.set("descr", label)
            document.add_paragraph(label, "Caption")
        elif kind == "link":
            link = document.add_paragraph()
            link.paragraph_format.space_after = Pt(2)
            link.paragraph_format.line_spacing = 1.05
            _link(link, *value)
    document.save(path)
