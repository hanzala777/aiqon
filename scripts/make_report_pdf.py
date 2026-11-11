"""Render reports/report.md -> reports/report.pdf (pure Python: markdown + fpdf2).

    python scripts/make_report_pdf.py
"""
import re
from pathlib import Path

import markdown
import matplotlib
from fpdf import FPDF, FontFace

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"  # DejaVu ships with matplotlib


def main():
    md = (ROOT / "reports" / "report.md").read_text(encoding="utf-8")
    html = markdown.markdown(md, extensions=["tables"])
    # fpdf2 resolves <img> paths relative to the CWD; make them absolute and size them.
    html = html.replace('src="figures/', f'src="{(ROOT / "reports" / "figures").as_posix()}/')
    html = html.replace("<img ", '<img width="520" ')
    html = html.replace("<table>", '<table width="100%" border="1" cellpadding="2">')
    html = html.replace("<code>", "<font face='DejaVuMono'>").replace("</code>", "</font>")
    # fpdf2 tables don't support inline tags inside cells -> strip them there.
    html = re.sub(r"(<t[dh](?:\s[^>]*)?>)(.*?)(</t[dh]>)",
                  lambda m: m.group(1) + re.sub(r"<[^>]+>", "", m.group(2)) + m.group(3), html, flags=re.S)

    pdf = FPDF(format="A4")
    pdf.set_margins(15, 12, 15)
    pdf.set_auto_page_break(True, margin=12)
    pdf.add_font("DejaVu", "", FONT_DIR / "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", FONT_DIR / "DejaVuSans-Bold.ttf")
    pdf.add_font("DejaVu", "I", FONT_DIR / "DejaVuSans-Oblique.ttf")
    pdf.add_font("DejaVu", "BI", FONT_DIR / "DejaVuSans-BoldOblique.ttf")
    pdf.add_font("DejaVuMono", "", FONT_DIR / "DejaVuSansMono.ttf")
    pdf.add_font("DejaVuMono", "I", FONT_DIR / "DejaVuSansMono-Oblique.ttf")
    pdf.add_font("DejaVuMono", "B", FONT_DIR / "DejaVuSansMono-Bold.ttf")
    pdf.add_font("DejaVuMono", "BI", FONT_DIR / "DejaVuSansMono-BoldOblique.ttf")
    pdf.add_page()
    pdf.set_font("DejaVu", size=8.2)
    navy = (20, 60, 100)
    styles = {"h1": FontFace(color=navy, size_pt=15, emphasis="B"),
              "h2": FontFace(color=navy, size_pt=10.5, emphasis="B")}
    pdf.write_html(html, font_family="DejaVu", tag_styles=styles,
                   li_prefix_color=(40, 40, 40), table_line_separators=True)
    out = ROOT / "reports" / "report.pdf"
    pdf.output(str(out))
    print(f"wrote {out} ({pdf.pages_count} pages)")


if __name__ == "__main__":
    main()
