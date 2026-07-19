"""Minimal markdown-subset -> PDF renderer using reportlab, with a
Cyrillic-capable font (Arial, from the system Windows fonts) registered so
Russian text renders correctly. Supports: #/##/### headers, - bullet lists,
**bold**, *italic*, blank-line-separated paragraphs, and *[...]* placeholder
emphasis (rendered in a distinct color so placeholders are visually obvious
to the reader, per the project rule to leave "заметный placeholder").
"""
import re
import sys

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_DIR = "C:/Windows/Fonts"
pdfmetrics.registerFont(TTFont("Arial", f"{FONT_DIR}/arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", f"{FONT_DIR}/arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", f"{FONT_DIR}/ariali.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", f"{FONT_DIR}/arialbi.ttf"))


SUBSCRIPT_DIGITS = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
})


def fix_unicode_subscripts(text):
    # Arial.ttf lacks glyphs for Unicode subscript digits (U+2080-2089), so
    # reportlab silently drops them (e.g. "mu_0(x)" -> "mu(x)"). Convert them
    # to reportlab's native <sub> markup instead, which renders correctly
    # regardless of font glyph coverage.
    return re.sub(
        "[₀-₉]+",
        lambda m: f"<sub>{m.group(0).translate(SUBSCRIPT_DIGITS)}</sub>",
        text,
    )


def inline_format(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = fix_unicode_subscripts(text)
    text = re.sub(r"\*\*(.+?)\*\*", r'<font face="Arial-Bold">\1</font>', text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r'<font face="Arial-Italic" color="#a05a00">\1</font>', text)
    return text


def build_styles():
    ss = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=18, leading=22, spaceAfter=8, spaceBefore=0, alignment=TA_LEFT),
        "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=13, spaceAfter=4, spaceBefore=9, textColor="#1a1a1a"),
        "h3": ParagraphStyle("h3", fontName="Arial-Bold", fontSize=10.5, spaceAfter=2, spaceBefore=6, textColor="#1a1a1a"),
        "body": ParagraphStyle("body", fontName="Arial", fontSize=9.5, leading=12.5, spaceAfter=5),
        "bullet": ParagraphStyle("bullet", fontName="Arial", fontSize=9.5, leading=12),
    }
    return styles


def markdown_to_flowables(md_text, styles):
    flowables = []
    lines = md_text.split("\n")
    i = 0
    bullet_buffer = []
    para_buffer = []

    def flush_bullets():
        nonlocal bullet_buffer
        if bullet_buffer:
            items = [ListItem(Paragraph(inline_format(b), styles["bullet"]), leftIndent=12) for b in bullet_buffer]
            flowables.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=14, spaceAfter=4))
            bullet_buffer = []

    def flush_para():
        nonlocal para_buffer
        if para_buffer:
            flowables.append(Paragraph(inline_format(" ".join(para_buffer)), styles["body"]))
            para_buffer = []

    def is_table_row(s):
        return s.strip().startswith("|") and s.strip().endswith("|")

    def is_separator_row(s):
        return re.match(r"^\|[\s:\-|]+\|$", s.strip()) is not None

    def parse_table_row(s):
        cells = s.strip().strip("|").split("|")
        return [c.strip() for c in cells]

    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            flush_bullets()
            flush_para()
            i += 1
            continue
        if is_table_row(line):
            flush_bullets(); flush_para()
            table_rows = []
            while i < len(lines) and is_table_row(lines[i]):
                if not is_separator_row(lines[i]):
                    table_rows.append(parse_table_row(lines[i]))
                i += 1
            cell_style = ParagraphStyle("cell", fontName="Arial", fontSize=8, leading=10)
            header_style = ParagraphStyle("cellhdr", fontName="Arial-Bold", fontSize=8.5, leading=10.5, textColor="#FFFFFF")
            data = []
            for r_idx, row in enumerate(table_rows):
                style = header_style if r_idx == 0 else cell_style
                data.append([Paragraph(inline_format(c), style) for c in row])
            n_cols = max(len(r) for r in data)
            avail_width = 6.4 * inch  # matches default body width for A4 with 0.7in margins
            col_w = avail_width / n_cols
            tbl = Table(data, colWidths=[col_w] * n_cols, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5EAA")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F3FA")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            flowables.append(tbl)
            flowables.append(Spacer(1, 8))
            continue
        if line.startswith("### "):
            flush_bullets(); flush_para()
            flowables.append(Paragraph(inline_format(line[4:]), styles["h3"]))
        elif line.startswith("## "):
            flush_bullets(); flush_para()
            flowables.append(Paragraph(inline_format(line[3:]), styles["h2"]))
        elif line.startswith("# "):
            flush_bullets(); flush_para()
            flowables.append(Paragraph(inline_format(line[2:]), styles["h1"]))
        elif line.strip().startswith("- "):
            flush_para()
            bullet_buffer.append(line.strip()[2:])
        elif bullet_buffer and line.startswith((" ", "\t")):
            # indented continuation of the previous bullet item (soft-wrapped
            # source line), not a new paragraph -- keeps list indentation intact
            bullet_buffer[-1] = bullet_buffer[-1] + " " + line.strip()
        else:
            flush_bullets()
            para_buffer.append(line.strip())
        i += 1
    flush_bullets(); flush_para()
    return flowables


def render(md_path, pdf_path, pagesize=A4, margins=0.7 * inch):
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    styles = build_styles()
    doc = SimpleDocTemplate(pdf_path, pagesize=pagesize,
                             leftMargin=margins, rightMargin=margins,
                             topMargin=margins, bottomMargin=margins)
    flowables = markdown_to_flowables(text, styles)
    doc.build(flowables)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
