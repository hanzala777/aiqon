"""Render reports/report.md -> reports/report.pdf (pure Python: matplotlib diagrams + fpdf2).

    python scripts/make_report_pdf.py            # redraws the diagrams, then builds the PDF
    python scripts/make_report_pdf.py --no-diagrams

Understands the Markdown used in report.md (headings, paragraphs, bullet lists, tables, images,
code blocks) plus three HTML comments that GitHub hides:
    <!-- center -->       centre the next paragraph
    <!-- small -->        set the next paragraph in small print
    <!-- width: 80 -->    next image at 80% of the text width
    <!-- cols: 20,50,30 -->  relative column widths of the next table
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import matplotlib
from PIL import Image
from fpdf import FPDF, FontFace
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)

BLACK, GREY = (20, 20, 20), (90, 90, 90)
BODY, LH = 9.5, 4.3       # body size (pt), line height (mm)
SMALL, SMALL_LH = 8.2, 3.6


def _fonts():
    """Palatino Linotype / Consolas when available (Windows), otherwise matplotlib's DejaVu fonts."""
    win = Path("C:/Windows/Fonts")
    serif = {"": win / "pala.ttf", "B": win / "palab.ttf", "I": win / "palai.ttf", "BI": win / "palabi.ttf"}
    mono = {"": win / "consola.ttf", "B": win / "consolab.ttf", "I": win / "consolai.ttf", "BI": win / "consolaz.ttf"}
    if all(p.exists() for p in [*serif.values(), *mono.values()]):
        return serif, mono
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    serif = {"": "DejaVuSerif.ttf", "B": "DejaVuSerif-Bold.ttf", "I": "DejaVuSerif-Italic.ttf",
             "BI": "DejaVuSerif-BoldItalic.ttf"}
    mono = {"": "DejaVuSansMono.ttf", "B": "DejaVuSansMono-Bold.ttf", "I": "DejaVuSansMono-Oblique.ttf",
            "BI": "DejaVuSansMono-BoldOblique.ttf"}
    return {k: d / v for k, v in serif.items()}, {k: d / v for k, v in mono.items()}


# ----------------------------------------------------------------------------- parser
BLOCK_START = re.compile(r"(#|\||!\[|[-*] |```|<!--)")


def parse(md: str) -> list[dict]:
    lines, blocks, pending, i = md.splitlines(), [], {}, 0

    def add(kind, **kw):
        nonlocal pending
        blocks.append({"kind": kind, "opts": pending, **kw})
        pending = {}

    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
        elif m := re.fullmatch(r"<!--\s*(.*?)\s*-->", s):
            key, _, val = m.group(1).partition(":")
            pending[key.strip()] = val.strip() or True
            i += 1
        elif s.startswith("```"):
            j = i + 1
            while not lines[j].strip().startswith("```"):
                j += 1
            add("code", text="\n".join(lines[i + 1:j]))
            i = j + 1
        elif s.startswith("#"):
            level = len(s) - len(s.lstrip("#"))
            add("h", level=level, text=s[level:].strip())
            i += 1
        elif s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            add("table", rows=[r for r in rows if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)])
        elif s.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", s)
            add("img", alt=m.group(1), path=m.group(2))
            i += 1
        elif re.match(r"[-*] ", s):
            items = []  # [level, text]; level 2 = indented sub-point
            while i < len(lines):
                ln, st = lines[i], lines[i].strip()
                if re.match(r"[-*] ", st) and not ln.startswith("  "):
                    items.append([1, st[2:]])
                elif re.match(r"[-*] ", st) and ln.startswith("  ") and items:
                    items.append([2, st[2:]])
                elif st and ln.startswith("  ") and items:
                    items[-1][1] += " " + st
                else:
                    break
                i += 1
            add("list", items=items)
        else:
            buf = []
            while i < len(lines) and lines[i].strip() and not BLOCK_START.match(lines[i].strip()):
                buf.append(lines[i].strip())
                i += 1
            text = " ".join(buf)
            if re.fullmatch(r"\*(Figure|Table) \d+.*\*", text):
                add("caption", text=text[1:-1])
            else:
                add("p", text=text)
    return blocks


def inline(text: str) -> str:
    """Markdown inline -> fpdf2 markdown (**bold**, __italic__, [link](url)); code spans become plain text."""
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return re.sub(r"(?<![\*\w])\*(?!\*)(.+?)(?<!\*)\*(?![\*\w])", r"__\1__", text)


# ----------------------------------------------------------------------------- PDF
class Report(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Serif", "", 8.5)
        self.set_text_color(*GREY)
        self.cell(0, 4, str(self.page_no()), align="C")

    def space_left(self):
        return self.page_break_trigger - self.get_y()

    def ensure(self, h):
        if self.space_left() < h:
            self.add_page()

    def write_text(self, text, size=BODY, lh=LH, style="", align="J", x=None, w=None, color=BLACK):
        x = self.l_margin if x is None else x
        self.set_font("Serif", style, size)
        self.set_text_color(*color)
        self.set_x(x)
        self.multi_cell(self.w - self.r_margin - x if w is None else w, lh, inline(text), markdown=True,
                        align=align, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def text_height(self, text, w, size=BODY, lh=LH, style=""):
        self.set_font("Serif", style, size)
        return self.multi_cell(w, lh, inline(text), markdown=True, dry_run=True, output="HEIGHT")


def image_height(pdf, block):
    """Printed height (mm) of an image block at its requested width."""
    w = pdf.epw * float(block["opts"].get("width", 100)) / 100
    with Image.open(REPORTS / block["path"]) as im:
        return w * im.height / im.width


def table_height(pdf, rows, widths, fs, lh, pad):
    colw = [pdf.epw * w / sum(widths) - 2 * pad[1] for w in widths]
    return sum(max(pdf.text_height(c, cw, size=fs, lh=lh, style="B" if n == 0 else "") for c, cw in zip(r, colw))
               + 2 * pad[0] for n, r in enumerate(rows))


def render(blocks, out: Path) -> int:
    serif, mono = _fonts()
    pdf = Report(format="A4")
    pdf.set_margins(19, 16, 19)
    pdf.set_auto_page_break(True, margin=16)
    for style, path in serif.items():
        pdf.add_font("Serif", style, path)
    for style, path in mono.items():
        pdf.add_font("Mono", style, path)
    pdf.set_title("Predicting Port Congestion and Shipment Delays")
    pdf.set_author("Hanzala Ahmed")
    pdf.add_page()

    for idx, b in enumerate(blocks):
        k, o = b["kind"], b["opts"]
        nxt = blocks[idx + 1] if idx + 1 < len(blocks) else {"kind": None}

        if k == "h" and b["level"] == 1:
            pdf.set_font("Serif", "B", 16)
            pdf.set_text_color(*BLACK)
            pdf.multi_cell(0, 7.5, b["text"], align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1.5)

        elif k == "h":
            if nxt["kind"] == "img":  # keep a heading on the same page as the figure under it
                pdf.ensure(10 + image_height(pdf, nxt) + 6)
            else:
                pdf.ensure(16)  # heading plus about two lines of text
            pdf.ln(2.4)
            pdf.set_font("Serif", "B", 11.2)
            pdf.set_text_color(*BLACK)
            pdf.multi_cell(0, 5.4, b["text"], align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1.0)

        elif k == "p":
            if o.get("center"):
                pdf.write_text(b["text"], align="C", color=GREY)
                if nxt["kind"] != "p" or not nxt["opts"].get("center"):
                    pdf.ln(3.5)
            elif o.get("small"):
                pdf.write_text(b["text"], size=SMALL, lh=SMALL_LH)
                pdf.ln(2.0)
            else:
                pdf.write_text(b["text"])
                pdf.ln(1.9)

        elif k == "list":
            for level, item in b["items"]:
                indent = 5 if level == 1 else 10
                pdf.ensure(min(pdf.text_height(item, pdf.epw - indent), 2 * LH))
                y = pdf.get_y()
                pdf.set_font("Serif", "", BODY)
                pdf.set_text_color(*BLACK)
                pdf.set_xy(pdf.l_margin + indent - 4, y)
                pdf.cell(4, LH, "\u2022" if level == 1 else "\u2013")
                pdf.write_text(item, x=pdf.l_margin + indent, align="L")
                pdf.ln(0.4)
            pdf.ln(1.2)

        elif k == "caption" and b["text"].startswith("Table"):
            pass  # drawn together with its table below

        elif k == "caption":
            pdf.write_text(b["text"], size=SMALL, lh=SMALL_LH, align="C" if len(b["text"]) < 110 else "J")
            pdf.ln(2.6)

        elif k == "table":
            rows = b["rows"]
            widths = [float(c) for c in o["cols"].split(",")] if "cols" in o else [1] * len(rows[0])
            fs, lh, pad = 8.6, 3.85, (0.9, 1.4)
            prev = blocks[idx - 1] if idx else {"kind": None}
            cap = prev["text"] if prev["kind"] == "caption" and prev["text"].startswith("Table") else None
            cap_h = pdf.text_height(cap, pdf.epw, size=SMALL, lh=SMALL_LH) + 1.5 if cap else 0
            pdf.ensure(cap_h + table_height(pdf, rows, widths, fs, lh, pad) + 2)
            if cap:
                pdf.write_text(cap, size=SMALL, lh=SMALL_LH, align="C")
                pdf.ln(1.5)
            pdf.set_draw_color(*BLACK)
            pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(*BLACK)
            pdf.set_font("Serif", "", fs)
            y0 = pdf.get_y()
            heading = FontFace(emphasis="BOLD", color=BLACK, size_pt=fs)
            with pdf.table(col_widths=widths, width=pdf.epw, line_height=lh, padding=pad, text_align="LEFT",
                           headings_style=heading, borders_layout="SINGLE_TOP_LINE", markdown=True,
                           v_align="M") as table:
                for r in rows:
                    row = table.row()
                    for c in r:
                        row.cell(inline(c))
            pdf.set_line_width(0.45)  # booktabs-style top and bottom rules
            pdf.line(pdf.l_margin, y0, pdf.w - pdf.r_margin, y0)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.set_line_width(0.2)
            pdf.ln(3.0)

        elif k == "img":
            path = REPORTS / b["path"]
            w = pdf.epw * float(o.get("width", 100)) / 100
            h = image_height(pdf, b)
            cap_h = (pdf.text_height(nxt["text"], pdf.epw, size=SMALL, lh=SMALL_LH)
                     if nxt["kind"] == "caption" else 0)
            pdf.ensure(h + 1.5 + cap_h)
            pdf.ln(0.5)
            pdf.image(str(path), x=pdf.l_margin + (pdf.epw - w) / 2, y=pdf.get_y(), w=w, h=h, alt_text=b["alt"])
            pdf.set_y(pdf.get_y() + h + 1.5)

        elif k == "code":
            lines = b["text"].splitlines()
            lh = 3.4
            h = len(lines) * lh + 3.2
            pdf.ensure(h + 2)
            y = pdf.get_y()
            pdf.set_fill_color(245, 245, 245)
            pdf.set_draw_color(200, 200, 200)
            pdf.set_line_width(0.2)
            pdf.rect(pdf.l_margin, y, pdf.epw, h, style="DF")
            pdf.set_font("Mono", "", 7.3)
            pdf.set_text_color(*BLACK)
            pdf.set_y(y + 1.6)
            for ln in lines:
                pdf.set_x(pdf.l_margin + 2.5)
                pdf.cell(pdf.epw - 5, lh, ln, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_y(y + h + 2.6)

    pdf.output(str(out))
    return pdf.pages_count


def main():
    if "--no-diagrams" not in sys.argv:
        import make_diagrams
        make_diagrams.main()
    out = REPORTS / "report.pdf"
    n = render(parse((REPORTS / "report.md").read_text(encoding="utf-8")), out)
    print(f"wrote {out} ({n} pages)")


if __name__ == "__main__":
    main()
