"""
plotstyle.py - one look for every figure in this repository.

Two things this module is strict about, and both are about the reader rather
than the plot.

Light and dark are two separate renders, not one render with the colours
flipped. Every figure is written twice, and the README picks between them with
a <picture> element, so the axes and labels are legible whichever theme GitHub
is showing. A single image with a transparent background looks fine in one
theme and unreadable in the other.

There is never a second y-axis. When two quantities share a plot box with
different scales, the reader has to work out which curve belongs to which axis
before they can read anything, and the relative heights of the two curves mean
nothing at all. Gain and phase go in stacked panels; so does everything else
that does not share units.

Series colours come from a palette whose adjacent pairs were checked for
colour-vision deficiency separation rather than chosen by eye, and every plot
with more than one series carries a legend, so colour is never the only thing
carrying identity.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIGHT = dict(
    surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7",
    series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
            "#008300", "#4a3aa7", "#e34948"],
)

DARK = dict(
    surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7", muted="#898781",
    grid="#2c2c2a", axis="#383835",
    series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
            "#008300", "#9085e9", "#e66767"],
)

THEMES = {"light": LIGHT, "dark": DARK}


def apply(theme):
    """Push one theme into matplotlib's rcParams."""
    c = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": c["surface"],
        "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"],
        "axes.edgecolor": c["axis"],
        "axes.labelcolor": c["secondary"],
        "axes.titlecolor": c["ink"],
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": c["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": c["ink"],
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "font.family": ["DejaVu Sans"],
        "font.size": 9,
        "figure.dpi": 110,
        "svg.fonttype": "none",
    })
    return c


def _round_coords(svg_text, places=1):
    """Round coordinates in an SVG to a sane number of decimal places.

    Matplotlib writes path coordinates at full float precision, which costs
    around twenty-five bytes for every point: "372.480000000000018" where
    "372.5" draws the same line at any size these figures are viewed. One
    decimal place is a tenth of a pixel, so nothing is lost that a screen or
    a printer could resolve.

    It is worth doing because GitHub stops rendering an SVG in a README above
    about 128 KB, and a figure nobody can see is worth nothing.
    """
    import re

    def fix(m):
        return f"{round(float(m.group(0)), places):g}"

    number = re.compile(r"-?\d+\.\d+")
    out, last = [], 0
    for attr in re.finditer(r'(?:\sd|\stransform|\spoints)="[^"]*"', svg_text):
        out.append(svg_text[last:attr.start()])
        out.append(number.sub(fix, attr.group(0)))
        last = attr.end()
    out.append(svg_text[last:])
    return "".join(out)


def save(fig, path_stem, theme):
    """Write one theme's SVG. Call once per theme with the same stem."""
    path = f"{path_stem}_{theme}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_round_coords(text))


def render(builder, path_stem):
    """Build and save the same figure in both themes.

    builder(colors) must construct and return a Figure using the colours it is
    handed, and must not reach for any colour of its own.
    """
    for theme in ("light", "dark"):
        c = apply(theme)
        fig = builder(c)
        save(fig, path_stem, theme)
    print(f"  wrote {path_stem}_light.svg and {path_stem}_dark.svg")


def annotate(ax, text, xy, xytext, c, arrow=True):
    """A callout in ink, never in a series colour.

    Text takes text colours so that a coloured mark beside a label carries
    identity and the words stay readable; a label painted in the series colour
    competes with the data and usually loses contrast.
    """
    ax.annotate(
        text, xy=xy, xytext=xytext, fontsize=8, color=c["secondary"],
        arrowprops=dict(arrowstyle="-", color=c["muted"], lw=0.7,
                        shrinkA=0, shrinkB=3) if arrow else None,
    )


def step_points(t, y):
    """Reduce a piecewise-constant signal to just its transitions.

    A switching waveform sampled every microsecond is forty thousand points
    per plot, almost all of which sit on top of each other along a flat top or
    a flat bottom. Keeping only the sample either side of each transition
    draws exactly the same picture from a few hundred points.

    This is lossless for a signal that only ever holds a level, which is what
    a pole voltage is. Nothing that varies smoothly, such as a load current,
    should be passed through it.
    """
    import numpy as np
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    change = np.flatnonzero(np.diff(y) != 0.0)
    keep = np.unique(np.concatenate([[0], change, change + 1, [len(y) - 1]]))
    return t[keep], y[keep]


def footnote(fig, text, c, width=104):
    """A wrapped caption under the figure.

    Wrapping matters more than it looks. The figure is saved with a tight
    bounding box, so one long unbroken line of caption widens the box and
    leaves the plot itself looking narrow and stranded in the middle of the
    image. Wrapping the caption to the plot's own width keeps the picture the
    widest thing in the file.
    """
    import textwrap
    # collapse runs of whitespace, but keep an explicit bullet separator
    # intact so a caption made of several measurements does not run together
    wrapped = "\n".join(textwrap.wrap(" ".join(text.split()), width=width))
    wrapped = wrapped.replace(" * ", "  -  ")
    fig.text(0.005, -0.012, wrapped, fontsize=8, color=c["secondary"],
             va="top")


def tidy(ax, c, legend=False, loc="upper right", ncol=1):
    """Recessive chrome: only the axes a reader needs."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(c["axis"])
    ax.spines["bottom"].set_color(c["axis"])
    if legend:
        ax.legend(loc=loc, ncol=ncol, labelcolor=c["secondary"],
                  handlelength=1.6, borderaxespad=0.4, columnspacing=1.2)
