"""Printable PDF reports (fpdf2 + matplotlib) and Excel export."""
from __future__ import annotations

import io
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fpdf import FPDF  # noqa: E402

_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
PALETTE = ["#1f4e79", "#c0504d", "#2e8b57", "#d4a017", "#6a3d9a", "#7f7f7f"]


def safe(txt) -> str:
    """Drop emoji / symbols the PDF font cannot draw (Greek letters are kept)."""
    return "".join(c for c in str(txt) if ord(c) < 0x2190 or 0x2200 <= ord(c) < 0x2300).strip()


def fmt_num(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "-"
        a = abs(v)
        if a == 0:
            return "0"
        if a >= 1e5:
            return f"{v:,.0f}"
        if a >= 1:
            return f"{v:,.2f}"
        return f"{v:.4f}"
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    return str(v)


class PDFReport:
    def __init__(self, title: str, subtitle: str = "", orientation: str = "L"):
        self.pdf = FPDF(orientation=orientation, unit="mm", format="A4")
        self.pdf.set_auto_page_break(True, margin=12)
        self.pdf.set_margins(12, 12, 12)
        reg = os.path.join(_FONT_DIR, "DejaVuSans.ttf")
        bold = os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")
        if os.path.exists(reg):
            self.pdf.add_font("DejaVu", "", reg)
            self.pdf.add_font("DejaVu", "B", bold if os.path.exists(bold) else reg)
            self.font = "DejaVu"
        else:  # pragma: no cover
            self.font = "Helvetica"
        self.pdf.add_page()
        self.pdf.set_font(self.font, "B", 16)
        self.pdf.cell(0, 9, safe(title), new_x="LMARGIN", new_y="NEXT")
        self.pdf.set_font(self.font, "", 9)
        self.pdf.set_text_color(90, 90, 90)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.pdf.cell(0, 5, safe(f"{subtitle}  |  Generated {stamp}  |  Option Analytics Lab").strip(" |"),
                      new_x="LMARGIN", new_y="NEXT")
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(3)

    @property
    def width(self):
        return self.pdf.w - self.pdf.l_margin - self.pdf.r_margin

    def heading(self, text: str):
        if self.pdf.get_y() > self.pdf.h - 60:
            self.pdf.add_page()
        self.pdf.ln(2)
        self.pdf.set_font(self.font, "B", 12)
        self.pdf.set_text_color(31, 78, 121)
        self.pdf.cell(0, 7, safe(text), new_x="LMARGIN", new_y="NEXT")
        self.pdf.set_text_color(0, 0, 0)

    def text(self, txt: str, size: int = 9):
        self.pdf.set_font(self.font, "", size)
        self.pdf.multi_cell(0, 4.6, safe(txt), new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(1)

    def kv(self, d: dict, cols: int = 2):
        items = [(safe(k), safe(fmt_num(v))) for k, v in d.items()]
        rows = [items[i:i + cols] for i in range(0, len(items), cols)]
        self.pdf.set_font(self.font, "", 9)
        with self.pdf.table(first_row_as_headings=False, line_height=5.2,
                            col_widths=tuple([3, 2] * cols)) as table:
            for row in rows:
                r = table.row()
                for k, v in row:
                    r.cell(k)
                    r.cell(v)
                for _ in range(cols - len(row)):
                    r.cell("")
                    r.cell("")
        self.pdf.ln(2)

    def table(self, df: pd.DataFrame, max_rows: int = 80, font_size: float | None = None):
        if df is None or df.empty:
            return
        d = df.head(max_rows)
        ncol = len(d.columns)
        fs = font_size or max(5.0, min(9.0, 150.0 / ncol))
        self.pdf.set_font(self.font, "", fs)
        with self.pdf.table(line_height=fs * 0.55, text_align="CENTER") as table:
            hdr = table.row()
            for c in d.columns:
                hdr.cell(safe(c))
            for rec in d.itertuples(index=False):
                row = table.row()
                for v in rec:
                    row.cell(safe(fmt_num(v)))
        if len(df) > max_rows:
            self.text(f"(first {max_rows} of {len(df)} rows shown - full data in the Excel export)", 7)
        self.pdf.ln(2)

    def figure(self, fig, width_ratio: float = 0.9):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        w = self.width * width_ratio
        h_est = w * fig.get_figheight() / fig.get_figwidth()
        if self.pdf.get_y() + h_est > self.pdf.h - 14:
            self.pdf.add_page()
        self.pdf.image(buf, x=self.pdf.l_margin + (self.width - w) / 2, w=w)
        self.pdf.ln(3)

    def output(self) -> bytes:
        return bytes(self.pdf.output())


# ---------------------------------------------------------------------------
# Matplotlib figures for reports
# ---------------------------------------------------------------------------
def line_fig(x, series: dict, title: str, xlabel: str = "", ylabel: str = "", markers=False,
             hline=None, vline=None, figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, y) in enumerate(series.items()):
        ax.plot(x, y, label=name, color=PALETTE[i % len(PALETTE)], lw=1.8, marker="o" if markers else None, ms=4)
    if hline is not None:
        ax.axhline(hline, color="#999", lw=0.8, ls="--")
    if vline is not None:
        ax.axvline(vline, color="#999", lw=0.8, ls=":")
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def scatter_line_fig(xs, ys, xl, yl, title, labels=("Market", "SABR fit"), xlabel="Strike", ylabel="Implied vol"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(xs, ys, color=PALETTE[1], s=22, label=labels[0], zorder=3)
    ax.plot(xl, yl, color=PALETTE[0], lw=2, label=labels[1])
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def hist_fig(data: dict, title: str, xlabel: str, bins=60):
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (name, x) in enumerate(data.items()):
        ax.hist(x, bins=bins, alpha=0.55, color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if len(data) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def to_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            if df is None:
                continue
            if isinstance(df, dict):
                df = pd.DataFrame({"Item": list(df.keys()), "Value": list(df.values())})
            df.to_excel(xw, sheet_name=str(name)[:31], index=False)
    return buf.getvalue()
