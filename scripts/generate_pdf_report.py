"""
generate_pdf_report.py
======================
Generates reports/final_report.pdf from reports/final_report.md
using only PyMuPDF (fitz >= 1.24) — no external dependencies needed.

Usage:
    python scripts/generate_pdf_report.py
"""

import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF (fitz) not found. Install with: pip install pymupdf")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "reports" / "final_report.md"
DST  = ROOT / "reports" / "final_report.pdf"

# ── Page geometry ─────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = fitz.paper_size("a4")   # 595 × 842 pts
MARGIN_L = 60
MARGIN_R = 60
MARGIN_T = 60
MARGIN_B = 60
TEXT_W   = PAGE_W - MARGIN_L - MARGIN_R

# ── Colours (r, g, b  in 0-1) ─────────────────────────────────────────────────
C_BLACK   = (0.10, 0.10, 0.10)
C_H1      = (0.07, 0.18, 0.40)
C_H2      = (0.10, 0.28, 0.55)
C_H3      = (0.14, 0.35, 0.60)
C_H4      = (0.20, 0.40, 0.65)
C_TABLE_H = (0.92, 0.95, 1.00)
C_TABLE_R = (0.98, 0.98, 0.98)
C_TABLE_B = (0.75, 0.82, 0.93)
C_CODE    = (0.94, 0.94, 0.94)
C_RULE    = (0.70, 0.78, 0.90)

# ── Font sizes & line heights ─────────────────────────────────────────────────
FS = {
    "h1": 18, "h2": 14, "h3": 12, "h4": 11,
    "body": 9.5, "small": 8.5, "code": 8.5,
    "table_h": 8.5, "table_r": 8,
}
LH = {k: v * 1.45 for k, v in FS.items()}   # line height = font_size × 1.45

# ── PDF document ──────────────────────────────────────────────────────────────
doc  = fitz.open()
page = doc.new_page(width=PAGE_W, height=PAGE_H)
y    = float(MARGIN_T)


def new_page():
    global page, y
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = float(MARGIN_T)


def ensure_space(needed: float):
    if y + needed > PAGE_H - MARGIN_B:
        new_page()


def insert_text(text: str, x: float, yy: float, fontsize: float,
                color=C_BLACK, fontname="helv", bold=False, italic=False):
    fn = fontname
    if bold and italic:
        fn = "hebo"     # Helvetica Bold Oblique — always embedded
    elif bold:
        fn = "hebo" if fontname == "helv" else fontname
    elif italic:
        fn = "hebi" if fontname == "helv" else fontname
    page.insert_text(
        (x, yy),
        text,
        fontname=fn,
        fontsize=fontsize,
        color=color,
    )


def text_width(text: str, fontsize: float, bold=False) -> float:
    fn = "hebo" if bold else "helv"
    return fitz.get_text_length(text, fontname=fn, fontsize=fontsize)


# ── Strip inline markdown (bold, italic, code, links) → plain text ────────────
def strip_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # links
    text = re.sub(r"`([^`]+)`", r"\1", text)               # code spans
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*",     r"\1", text)
    text = re.sub(r"\*(.+?)\*",         r"\1", text)
    text = re.sub(r"_(.+?)_",           r"\1", text)
    text = re.sub(r"~~(.+?)~~",         r"\1", text)
    text = re.sub(r"<!--.*?-->",        "",    text)
    # Simplify TeX math markers
    text = re.sub(r"\$\$?([^$]+)\$\$?", r"\1", text)
    return text.strip()


# ── Word-wrap a plain string into lines of max pixel width w ──────────────────
def wrap(text: str, max_w: float, fontsize: float, bold=False) -> list:
    words  = text.split()
    lines  = []
    current = ""
    for w in words:
        trial = (current + " " + w).strip()
        if text_width(trial, fontsize, bold) <= max_w:
            current = trial
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines if lines else [""]


# ── Draw a horizontal rule ────────────────────────────────────────────────────
def draw_rule(thickness=0.5):
    global y
    ensure_space(6)
    page.draw_line(
        (MARGIN_L, y + 3), (PAGE_W - MARGIN_R, y + 3),
        color=C_RULE, width=thickness,
    )
    y += 8


# ── Render one wrapped paragraph ──────────────────────────────────────────────
def render_paragraph(text: str, indent=0, fontsize=None, color=None,
                     bold=False, italic=False, extra_before=0, extra_after=2):
    global y
    if fontsize is None:
        fontsize = FS["body"]
    if color is None:
        color = C_BLACK
    lh = fontsize * 1.45
    plain = strip_inline(text)
    if not plain:
        return
    lines = wrap(plain, TEXT_W - indent, fontsize, bold)
    if extra_before:
        y += extra_before
    for ln in lines:
        ensure_space(lh)
        insert_text(ln, MARGIN_L + indent, y + fontsize,
                    fontsize, color, bold=bold, italic=italic)
        y += lh
    y += extra_after


# ── Render a markdown table ───────────────────────────────────────────────────
def render_table(rows: list):
    """rows: list of list-of-strings (first row = header)."""
    global y
    if not rows:
        return

    n_cols = max(len(r) for r in rows)
    # Pad rows
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    col_w_raw = [0.0] * n_cols
    for row in rows:
        for ci, cell in enumerate(row):
            col_w_raw[ci] = max(col_w_raw[ci],
                                text_width(strip_inline(cell), FS["table_r"], False) + 8)
    total = sum(col_w_raw)
    scale = min(1.0, TEXT_W / total)
    col_w = [w * scale for w in col_w_raw]

    row_h = FS["table_r"] * 1.6
    hdr_h = FS["table_h"] * 1.7

    for ri, row in enumerate(rows):
        is_hdr = (ri == 0)
        is_sep = all(set(c.strip()) <= {"-", ":"} for c in row if c.strip())
        if is_sep:
            continue

        rh = hdr_h if is_hdr else row_h
        ensure_space(rh + 2)

        # Row background
        bg = C_TABLE_H if is_hdr else (C_TABLE_R if ri % 2 == 0 else (1, 1, 1))
        x0 = MARGIN_L
        page.draw_rect(fitz.Rect(x0, y, x0 + sum(col_w), y + rh),
                       color=None, fill=bg)

        # Top border for header
        if is_hdr:
            page.draw_line((x0, y), (x0 + sum(col_w), y),
                           color=C_TABLE_B, width=1.0)

        # Cell text
        cx = x0
        for ci, cell in enumerate(row):
            plain = strip_inline(cell)
            page.insert_text(
                (cx + 4, y + (FS["table_h"] if is_hdr else FS["table_r"]) + 2),
                plain[:60],   # truncate very long cells
                fontname="hebo" if is_hdr else "helv",
                fontsize=FS["table_h"] if is_hdr else FS["table_r"],
                color=C_H2 if is_hdr else C_BLACK,
            )
            # Vertical separator
            page.draw_line((cx, y), (cx, y + rh),
                           color=C_TABLE_B, width=0.3)
            cx += col_w[ci]

        # Bottom border
        page.draw_line((x0, y + rh), (x0 + sum(col_w), y + rh),
                       color=C_TABLE_B, width=0.3)
        y += rh

    y += 6


# ── Parse markdown rows ────────────────────────────────────────────────────────
def parse_table_row(line: str) -> list:
    line = line.strip().strip("|")
    return [c.strip() for c in line.split("|")]


# ── Main renderer ─────────────────────────────────────────────────────────────
def render_md(md_text: str):
    global y
    lines  = md_text.splitlines()
    i      = 0
    table_buf: list = []

    def flush_table():
        nonlocal table_buf
        if table_buf:
            render_table(table_buf)
            table_buf = []

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        # ── Skip YAML / HTML comments ──────────────────────────────────────
        if stripped.startswith("<!--") or stripped == "---" and i == 0:
            i += 1
            continue

        # ── Horizontal rule ────────────────────────────────────────────────
        if re.match(r"^---+$", stripped) or re.match(r"^\*\*\*+$", stripped):
            flush_table()
            draw_rule()
            i += 1
            continue

        # ── Table row ─────────────────────────────────────────────────────
        if stripped.startswith("|"):
            table_buf.append(parse_table_row(stripped))
            i += 1
            continue
        else:
            flush_table()

        # ── Blank line ─────────────────────────────────────────────────────
        if stripped == "":
            y += 4
            i += 1
            continue

        # ── ATX Headings ───────────────────────────────────────────────────
        m = re.match(r"^(#{1,4})\s+(.*)", stripped)
        if m:
            level  = len(m.group(1))
            text   = strip_inline(m.group(2))
            keys   = ["h1", "h2", "h3", "h4"]
            key    = keys[level - 1]
            colors = [C_H1, C_H2, C_H3, C_H4]
            fs     = FS[key]
            col    = colors[level - 1]
            before = [14, 10, 7, 5][level - 1]
            y += before
            ensure_space(fs * 2)
            # Background band for H1/H2
            if level == 1:
                page.draw_rect(
                    fitz.Rect(MARGIN_L - 4, y - 2, PAGE_W - MARGIN_R + 4, y + fs + 6),
                    color=None, fill=(0.07, 0.18, 0.40),
                )
                page.insert_text((MARGIN_L, y + fs),
                                 text, fontname="hebo", fontsize=fs,
                                 color=(1, 1, 1))
            elif level == 2:
                page.draw_rect(
                    fitz.Rect(MARGIN_L - 4, y - 1, PAGE_W - MARGIN_R + 4, y + fs + 4),
                    color=None, fill=C_TABLE_H,
                )
                page.draw_line((MARGIN_L - 4, y + fs + 4),
                               (PAGE_W - MARGIN_R + 4, y + fs + 4),
                               color=C_H2, width=1.2)
                page.insert_text((MARGIN_L, y + fs),
                                 text, fontname="hebo", fontsize=fs, color=col)
            else:
                page.insert_text((MARGIN_L, y + fs),
                                 text, fontname="hebo", fontsize=fs, color=col)
            y += fs + 6
            i += 1
            continue

        # ── Fenced code block ──────────────────────────────────────────────
        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1   # closing fence
            for cl in code_lines:
                plain = cl.rstrip()
                fs = FS["code"]
                lh = fs * 1.35
                ensure_space(lh + 1)
                page.draw_rect(
                    fitz.Rect(MARGIN_L - 2, y, PAGE_W - MARGIN_R + 2, y + lh),
                    color=None, fill=C_CODE,
                )
                page.insert_text((MARGIN_L + 4, y + fs),
                                 plain[:120],
                                 fontname="cour", fontsize=fs, color=C_BLACK)
                y += lh
            y += 4
            continue

        # ── Bullet / unordered list ────────────────────────────────────────
        m_bullet = re.match(r"^(\s*)[*\-+]\s+(.*)", raw)
        if m_bullet:
            indent_spaces = len(m_bullet.group(1))
            indent_pt = MARGIN_L + 8 + indent_spaces * 4
            text_raw = m_bullet.group(2)
            plain = strip_inline(text_raw)
            fs = FS["body"]
            lh = fs * 1.45
            lines_w = wrap(plain, TEXT_W - 16 - indent_spaces * 4, fs)
            ensure_space(lh * len(lines_w))
            # Bullet dot
            page.insert_text((indent_pt - 10, y + fs),
                             "•", fontname="helv", fontsize=fs, color=C_H2)
            for li, ln in enumerate(lines_w):
                ensure_space(lh)
                page.insert_text((indent_pt, y + fs),
                                 ln, fontname="helv", fontsize=fs, color=C_BLACK)
                y += lh
            i += 1
            continue

        # ── Numbered list ──────────────────────────────────────────────────
        m_num = re.match(r"^\s*(\d+)\.\s+(.*)", stripped)
        if m_num:
            num  = m_num.group(1)
            text_raw = m_num.group(2)
            plain = strip_inline(text_raw)
            fs = FS["body"]
            lh = fs * 1.45
            wrapped = wrap(plain, TEXT_W - 20, fs)
            ensure_space(lh * len(wrapped))
            page.insert_text((MARGIN_L, y + fs),
                             f"{num}.", fontname="hebo",
                             fontsize=fs, color=C_H2)
            for li, ln in enumerate(wrapped):
                ensure_space(lh)
                page.insert_text((MARGIN_L + 18, y + fs),
                                 ln, fontname="helv", fontsize=fs, color=C_BLACK)
                y += lh
            i += 1
            continue

        # ── Blockquote ────────────────────────────────────────────────────
        if stripped.startswith(">"):
            text_raw = stripped.lstrip(">").strip()
            plain = strip_inline(text_raw)
            fs = FS["small"]
            lh = fs * 1.4
            wrapped = wrap(plain, TEXT_W - 20, fs)
            ensure_space(lh * len(wrapped) + 4)
            page.draw_line((MARGIN_L + 2, y),
                           (MARGIN_L + 2, y + lh * len(wrapped) + 2),
                           color=C_H3, width=2)
            for ln in wrapped:
                ensure_space(lh)
                page.insert_text((MARGIN_L + 10, y + fs),
                                 ln, fontname="hebi", fontsize=fs,
                                 color=(0.3, 0.3, 0.3))
                y += lh
            y += 4
            i += 1
            continue

        # ── Normal paragraph ───────────────────────────────────────────────
        plain = strip_inline(stripped)
        if plain:
            render_paragraph(plain, fontsize=FS["body"],
                             extra_before=1, extra_after=3)
        i += 1

    flush_table()


# ── Cover page ────────────────────────────────────────────────────────────────
def make_cover():
    global y
    # Dark header band
    page.draw_rect(fitz.Rect(0, 0, PAGE_W, 200), color=None,
                   fill=C_H1)
    page.insert_text((MARGIN_L, 80),
                     "Faithfulness-Aware Explainability Audit",
                     fontname="hebo", fontsize=16, color=(1, 1, 1))
    page.insert_text((MARGIN_L, 105),
                     "of Multilingual Transformer Models",
                     fontname="hebo", fontsize=16, color=(1, 1, 1))
    page.insert_text((MARGIN_L, 130),
                     "for Code-Mixed Hate Speech Detection",
                     fontname="hebo", fontsize=16, color=(1, 1, 1))
    page.draw_line((MARGIN_L, 150), (PAGE_W - MARGIN_R, 150),
                   color=(0.5, 0.7, 1.0), width=1.5)
    page.insert_text((MARGIN_L, 170),
                     "Team 6 — XAI Research Project | Final Report | October 2026",
                     fontname="helv", fontsize=10, color=(0.8, 0.9, 1.0))
    y = 220
    meta = [
        ("Repository:",     "Team6_XAI"),
        ("Base Model:",     "bert-base-multilingual-cased (mBERT)"),
        ("Datasets:",       "HECM · Tamil DravidianCodeMix · Malayalam DravidianCodeMix"),
        ("Total Samples:",  "73,286"),
        ("Execution Seed:", "42"),
        ("Audit Dimensions:",
         "Predictive Performance · Calibration · Explainability · Robustness"),
        ("Status:",         "Final — All results are authoritative Colab artifacts"),
    ]
    for label, val in meta:
        page.insert_text((MARGIN_L, y),
                         label, fontname="hebo", fontsize=9.5, color=C_H2)
        page.insert_text((MARGIN_L + 130, y),
                         val,   fontname="helv", fontsize=9.5, color=C_BLACK)
        y += 16
    draw_rule()


# ── Page numbers ──────────────────────────────────────────────────────────────
def add_page_numbers():
    for pno in range(len(doc)):
        pg = doc[pno]
        if pno == 0:
            continue   # no number on cover
        pg.insert_text(
            (PAGE_W / 2 - 10, PAGE_H - 20),
            str(pno),
            fontname="helv", fontsize=8, color=(0.5, 0.5, 0.5),
        )


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Reading: {SRC}")
    if not SRC.exists():
        print("ERROR: reports/final_report.md not found.")
        sys.exit(1)

    md_text = SRC.read_text(encoding="utf-8")

    # Cover page
    make_cover()

    # Skip the first H1 (already on cover) and the abstract header preamble
    # Start rendering from line 1 (we skip the very first line which is the title)
    lines = md_text.splitlines()
    # Find where abstract starts (skip cover title block)
    start = 0
    for idx, ln in enumerate(lines):
        if ln.strip().startswith("## Abstract"):
            start = idx
            break
    body_text = "\n".join(lines[start:])

    # New page for body
    new_page()
    render_md(body_text)

    add_page_numbers()

    doc.save(str(DST), garbage=4, deflate=True)
    print(f"PDF saved:  {DST}  ({DST.stat().st_size:,} bytes, {len(doc)} pages)")
