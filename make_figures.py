#!/usr/bin/env python3
"""
make_figures.py - regenerates every figure in figures/ from the model.

Nothing here is drawn by hand. Run the script and each figure is rebuilt from
the design equations and the simulation, so a change to the converter or the
compensator shows up in the pictures instead of quietly disagreeing with them.

Every figure is written twice, once for a light page and once for a dark one.
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from models.buck_blocks import BuckParams
from models.buck_model import (load_step_metrics, simulate, startup_metrics)
from models.design import (closed_loop_poles, design, loop_response, margins,
                           plant_response, to_discrete)
from models.plotstyle import annotate, footnote, render, tidy

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

P = BuckParams(n_dpwm=12)

# The three compensators the project compares. The middle one exists to be
# wrong: it is what happens when a type II design is pushed for bandwidth it
# cannot deliver on a lightly damped output filter.
DESIGNS = [
    ("type II, 200 Hz", 2, 200.0, 60.0),
    ("type II, 600 Hz", 2, 600.0, 60.0),
    ("type III, 4 kHz", 3, 4000.0, 50.0),
]


def build_designs():
    out = []
    for label, order, fc, pm in DESIGNS:
        (num, den), rec = design(P, f_cross=fc, pm_deg=pm, order=order)
        b, a = to_discrete(num, den, P.ts, f_prewarp=fc)
        m = margins(P, num, den)
        cl = closed_loop_poles(P, b, a)
        out.append(dict(label=label, order=order, fc=fc, pm=pm,
                        num=num, den=den, b=b, a=a, rec=rec,
                        margins=m, cl=cl))
    return out


# ----------------------------------------------------------------------
def fig_plant(P):
    """The plant on its own, and where its phase goes."""

    def build(c):
        f = np.logspace(1, np.log10(0.5 / P.ts), 3000)
        H = plant_response(P, f)
        H_nodelay = plant_response(P, f) * np.exp(1j * 2 * np.pi * f * 1.5 * P.ts)

        fig, ax = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
        ax[0].semilogx(f, 20 * np.log10(np.abs(H)), lw=1.9, color=c["series"][0])
        ax[0].set_ylabel("magnitude (dB)")
        ax[0].set_title("Control-to-output response of the power stage",
                        loc="left")

        ax[1].semilogx(f, np.degrees(np.unwrap(np.angle(H_nodelay))), lw=1.9,
                       ls=(0, (5, 3)), color=c["series"][2],
                       label="power stage alone")
        ax[1].semilogx(f, np.degrees(np.unwrap(np.angle(H))), lw=1.9,
                       color=c["series"][0], label="with the loop's 1.5-sample delay")
        ax[1].set_ylabel("phase (degrees)")
        ax[1].set_xlabel("frequency (Hz)")
        ax[1].axhline(-180, lw=0.9, ls=(0, (4, 3)), color=c["axis"])
        ax[1].legend(loc="lower left", labelcolor=c["secondary"],
                     handlelength=2.2, borderaxespad=0.4)
        ax[1].set_title("Phase, with and without the delay", loc="left")

        for a_ in ax:
            a_.axvline(P.f_lc, lw=0.9, ls=(0, (2, 3)), color=c["muted"])
            tidy(a_, c)
        annotate(ax[0], f"LC corner {P.f_lc:.0f} Hz, Q = {P.q_lc:.1f}\n"
                        f"the peak is why a PI cannot cope here",
                 xy=(P.f_lc, 20 * np.log10(np.max(np.abs(H)))),
                 xytext=(P.f_lc * 1.5, 20 * np.log10(np.max(np.abs(H))) - 3), c=c)

        fig.suptitle("Buck power stage, 24 V to 12 V, 100 kHz",
                     y=0.998, fontsize=11, color=c["ink"], ha="left", x=0.005)
        fig.tight_layout()
        footnote(fig,
                 "The delay costs nothing in gain and a great deal in phase, "
                 "and the loss grows in proportion to frequency: about 5 "
                 "degrees at 1 kHz and 54 degrees at a tenth of the switching "
                 "frequency. It is the reason a digital loop cannot simply be "
                 "designed as though it were an analogue one.", c)
        return fig

    render(build, os.path.join(OUT, "plant_bode"))


def fig_loop(designs):
    """Loop gain for all three compensators, with the margins marked."""

    def build(c):
        f = np.logspace(1, np.log10(0.5 / P.ts), 4000)
        fig, ax = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
        styles = ["-", (0, (5, 3)), "-"]

        for i, d in enumerate(designs):
            T = loop_response(P, d["num"], d["den"], f)
            lab = (f"{d['label']}  "
                   f"{'stable' if d['cl']['stable'] else 'UNSTABLE'}")
            ax[0].semilogx(f, 20 * np.log10(np.abs(T)), lw=1.9,
                           ls=styles[i], color=c["series"][i], label=lab)
            ax[1].semilogx(f, np.degrees(np.unwrap(np.angle(T))), lw=1.9,
                           ls=styles[i], color=c["series"][i])
            if np.isfinite(d["margins"]["f_cross"]):
                ax[0].plot([d["margins"]["f_cross"]], [0.0], "o", ms=7,
                           color=c["series"][i], mec=c["surface"], mew=1.6,
                           zorder=5)

        ax[0].axhline(0, lw=0.9, color=c["axis"])
        ax[0].set_ylabel("loop gain (dB)")
        ax[0].set_ylim(-60, 90)
        ax[0].set_title("Open-loop gain; circles mark each gain crossover",
                        loc="left")
        ax[0].legend(loc="upper right", labelcolor=c["secondary"],
                     handlelength=2.2, borderaxespad=0.3)

        ax[1].axhline(-180, lw=0.9, ls=(0, (4, 3)), color=c["axis"])
        ax[1].set_ylabel("phase (degrees)")
        ax[1].set_xlabel("frequency (Hz)")
        ax[1].set_title("Phase; the distance above the dashed line at "
                        "crossover is the phase margin", loc="left")

        for a_ in ax:
            tidy(a_, c)

        fig.suptitle("Three compensators on the same power stage",
                     y=0.998, fontsize=11, color=c["ink"], ha="left", x=0.005)
        fig.tight_layout()
        rows = " * ".join(
            f"{d['label']}: PM {d['margins']['pm']:.0f} deg, "
            f"GM {d['margins']['gm_db']:.1f} dB" for d in designs)
        footnote(fig, rows + ".  The 600 Hz design crosses 0 dB three times, "
                 "because the resonant peak pushes the gain back above unity. "
                 "Phase margin is defined at one crossing only, so on this "
                 "loop it is not a stability test - the pole map is.", c)
        return fig

    render(build, os.path.join(OUT, "loop_bode"))


def fig_poles(designs):
    """Closed-loop poles in the z-plane: the test that actually decides."""

    def build(c):
        # Two panels rather than one with an inset. The whole result lives in
        # the fourth decimal place: at full scale a pole at 1.0018 and one at
        # 0.9930 are the same dot, so the zoom is not decoration - without it
        # the figure cannot show what it claims.
        fig, (ax, zx) = plt.subplots(1, 2, figsize=(9.0, 4.6))
        th = np.linspace(0, 2 * np.pi, 800)
        markers = ["o", "s", "D"]

        ax.plot(np.cos(th), np.sin(th), lw=1.3, color=c["axis"])
        ax.axhline(0, lw=0.8, color=c["grid"])
        for i, d in enumerate(designs):
            p_ = d["cl"]["poles"]
            ax.plot(p_.real, p_.imag, markers[i], ms=9, mfc="none", mew=1.9,
                    color=c["series"][i],
                    label=f"{d['label']}  max |z| = "
                          f"{d['cl']['spectral_radius']:.4f}")
        ax.set_aspect("equal")
        tidy(ax, c)

        # The right-hand panel drops the argument and plots only the magnitude,
        # because magnitude is the entire question. A zoom of the z-plane near
        # z = 1 would have to be stretched to fit the poles in, and a stretched
        # unit circle is no longer a boundary anyone can read by eye.
        for i, d in enumerate(designs):
            mags = np.abs(d["cl"]["poles"])
            zx.plot(mags, np.full_like(mags, i, dtype=float), markers[i],
                    ms=9, mfc="none", mew=1.9, color=c["series"][i])
        zx.axvline(1.0, lw=1.2, color=c["axis"])
        zx.axvspan(1.0, 1.02, color=c["series"][1], alpha=0.09, lw=0)
        zx.set_yticks(range(len(designs)))
        zx.set_yticklabels([d["label"] for d in designs], fontsize=8)
        zx.set_xlim(0.955, 1.012)
        zx.set_ylim(-0.6, len(designs) - 0.4)
        zx.set_xlabel("pole magnitude  |z|")
        zx.set_title("Only the magnitude matters: 1 is the boundary",
                     loc="left")
        zx.text(1.006, len(designs) - 0.55, "unstable", fontsize=8,
                ha="center", color=c["secondary"])
        for i, d in enumerate(designs):
            zx.text(d["cl"]["spectral_radius"], i + 0.22,
                    f"{d['cl']['spectral_radius']:.4f}", fontsize=8,
                    ha="center", color=c["secondary"])
        tidy(zx, c)

        ax.axvline(0, lw=0.8, color=c["grid"])
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_xlabel("real")
        ax.set_ylabel("imaginary")
        ax.set_title("All closed-loop poles", loc="left")

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1,
                  labelcolor=c["secondary"], handlelength=1.4, fontsize=8)

        fig.suptitle("The stability test that decides",
                     y=0.995, fontsize=11, color=c["ink"], ha="left", x=0.005)
        fig.tight_layout()
        footnote(fig,
                 "Inside the circle is stable, outside is not, and the marker "
                 "shapes carry the same identity as the colours. The 600 Hz "
                 "design sits just outside at 1.0018, which grows so slowly "
                 "that a short simulation looks settled: a twelve-millisecond "
                 "run of it converges, and a sixty-millisecond run does not.",
                 c, width=88)
        return fig

    render(build, os.path.join(OUT, "pole_map"))


def fig_transient(designs):
    """Start-up and a load step, the two things a supply is judged on."""
    runs = []
    for d in designs:
        r = simulate(P, d["b"], d["a"], t_end=14e-3, dt=4e-7, t_ramp=1e-3,
                     load_step=(6e-3, 3.0), averaged=True)
        runs.append((d, r))

    def build(c):
        fig, ax = plt.subplots(2, 1, figsize=(7.2, 6.6))
        styles = ["-", (0, (5, 3)), "-"]

        for i, (d, r) in enumerate(runs):
            ax[0].plot(r["t"] * 1e3, r["vout"], lw=1.8, ls=styles[i],
                       color=c["series"][i], label=d["label"])
            m = (r["t"] >= 5.8e-3) & (r["t"] <= 8.5e-3)
            ax[1].plot((r["t"][m] - 6e-3) * 1e3, r["vout"][m], lw=1.8,
                       ls=styles[i], color=c["series"][i], label=d["label"])

        ax[0].axhline(P.vref, lw=0.9, ls=(0, (4, 3)), color=c["axis"])
        ax[0].set_ylabel("$V_{out}$  (V)")
        ax[0].set_xlabel("time (ms)")
        ax[0].set_ylim(0, 26)
        ax[0].set_title("Start-up on a 1 ms reference ramp, then a 2 A to 4 A "
                        "load step at 6 ms", loc="left")
        ax[0].legend(loc="lower right", labelcolor=c["secondary"],
                     handlelength=2.2, borderaxespad=0.4)

        ax[1].axhline(P.vref, lw=0.9, ls=(0, (4, 3)), color=c["axis"])
        for s in (0.99, 1.01):
            ax[1].axhline(P.vref * s, lw=0.8, ls=(0, (2, 3)), color=c["muted"])
        ax[1].set_ylabel("$V_{out}$  (V)")
        ax[1].set_xlabel("time after the load step (ms)")
        ax[1].set_title("The same step, close up; dotted lines are the "
                        "1 % band", loc="left")

        for a_ in ax:
            tidy(a_, c)

        fig.suptitle("What the compensator choice costs at the output",
                     y=0.998, fontsize=11, color=c["ink"], ha="left", x=0.005)
        fig.tight_layout()

        parts = []
        for d, r in runs:
            st = load_step_metrics(r["t"], r["vout"], 6e-3, P.vref, window=6e-3)
            parts.append(f"{d['label']}: dip {st['deviation_pct']:.1f} %, "
                         f"back inside 1 % after {st['recovery_s']*1e3:.2f} ms")
        footnote(fig, " * ".join(parts) +
                 ".  Twenty times the bandwidth buys a shallower dip and a "
                 "recovery several times faster, which is what the extra two "
                 "poles and two zeros of the type III are actually for.", c)
        return fig

    render(build, os.path.join(OUT, "transient"))


def fig_ripple(designs):
    """Switching model against averaged model, plus the ripple the datasheet quotes."""
    d = designs[-1]
    rs = simulate(P, d["b"], d["a"], t_end=500e-6, dt=2e-8, t_ramp=0.0,
                  settled=True, averaged=False)
    ra = simulate(P, d["b"], d["a"], t_end=500e-6, dt=2e-7, t_ramp=0.0,
                  settled=True, averaged=True)

    def build(c):
        t0, t1 = 400e-6, 430e-6
        m = (rs["t"] >= t0) & (rs["t"] <= t1)
        ma = (ra["t"] >= t0) & (ra["t"] <= t1)
        tm = (rs["t"][m] - t0) * 1e6
        tma = (ra["t"][ma] - t0) * 1e6

        fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True)

        ax[0].plot(tm, rs["gate"][m], lw=1.2, color=c["series"][0])
        ax[0].set_ylabel("gate")
        ax[0].set_ylim(-0.15, 1.35)
        ax[0].set_title("High-side gate signal, 100 kHz", loc="left")

        ax[1].plot(tm, rs["il"][m], lw=1.6, color=c["series"][1])
        ax[1].set_ylabel("$i_L$  (A)")
        ax[1].set_title("Inductor current: the triangle the averaged model "
                        "replaces with its mean", loc="left")

        ax[2].plot(tm, rs["vout"][m] * 1e3 - P.vref * 1e3, lw=1.6,
                   color=c["series"][2], label="switching model")
        ax[2].plot(tma, ra["vout"][ma] * 1e3 - P.vref * 1e3, lw=2.0,
                   ls=(0, (5, 3)), color=c["muted"], label="averaged model")
        ax[2].set_ylabel("$V_{out} - 12$ V  (mV)")
        ax[2].set_xlabel("time (microseconds)")
        ax[2].set_title("Output ripple, and the averaged model through the "
                        "middle of it", loc="left")
        # headroom above the ripple, so the legend cannot sit on the trace
        lo, hi = ax[2].get_ylim()
        ax[2].set_ylim(lo, hi + 0.55 * (hi - lo))
        ax[2].legend(loc="upper right", ncol=2, labelcolor=c["secondary"],
                     handlelength=2.2, borderaxespad=0.3, columnspacing=1.4)

        for a_ in ax:
            tidy(a_, c)

        fig.suptitle("The switching model, and the averaged model it has to agree with",
                     y=0.998, fontsize=11, color=c["ink"], ha="left", x=0.005)
        fig.tight_layout()

        w = rs["t"] > 400e-6
        il_pp = float(np.max(rs["il"][w]) - np.min(rs["il"][w]))
        vo_pp = float(np.max(rs["vout"][w]) - np.min(rs["vout"][w]))
        esr = P.rc * P.ripple_current()
        cap = P.ripple_current() / (8 * P.C * P.fsw)
        footnote(fig,
                 f"Inductor ripple {il_pp:.3f} A against "
                 f"{P.ripple_current():.3f} A from volt-seconds across L. "
                 f"Output ripple {vo_pp*1e3:.1f} mV against an upper bound of "
                 f"{(esr+cap)*1e3:.1f} mV, of which {esr*1e3:.0f} mV is the "
                 f"capacitor's ESR and only {cap*1e3:.1f} mV is its "
                 f"capacitance. On a real board the ESR is the specification "
                 f"that matters.", c)
        return fig

    render(build, os.path.join(OUT, "ripple"))


def fig_windup(designs):
    """Why anti-windup is not optional."""
    d = designs[-1]
    sag = [(0.0, 24.0), (4e-3, 10.0), (10e-3, 24.0)]
    runs = {}
    for aw in (True, False):
        runs[aw] = simulate(P, d["b"], d["a"], t_end=18e-3, dt=4e-7,
                            t_ramp=1e-3, averaged=True, antiwindup=aw,
                            vin_profile=sag)

    def build(c):
        fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.8), sharex=True)
        r_on, r_off = runs[True], runs[False]

        ax[0].plot(r_on["t"] * 1e3, r_on["vin"], lw=1.8, color=c["muted"])
        ax[0].set_ylabel("$V_{in}$  (V)")
        ax[0].set_ylim(0, 27)
        ax[0].set_title("Input sags to 10 V, which is below what the "
                        "converter needs to hold 12 V", loc="left")

        ax[1].plot(r_off["t"] * 1e3, r_off["vout"], lw=1.8, ls=(0, (5, 3)),
                   color=c["series"][1], label="no anti-windup")
        ax[1].plot(r_on["t"] * 1e3, r_on["vout"], lw=1.8,
                   color=c["series"][2], label="with anti-windup")
        ax[1].axhline(P.vref, lw=0.9, ls=(0, (4, 3)), color=c["axis"])
        ax[1].set_ylabel("$V_{out}$  (V)")
        ax[1].set_title("Output. Both ride the sag down; only one comes back "
                        "cleanly", loc="left")
        ax[1].legend(loc="upper left", labelcolor=c["secondary"],
                     handlelength=2.2, borderaxespad=0.4)

        ax[2].plot(r_off["t"] * 1e3, r_off["duty"], lw=1.8, ls=(0, (5, 3)),
                   color=c["series"][1], label="no anti-windup")
        ax[2].plot(r_on["t"] * 1e3, r_on["duty"], lw=1.8,
                   color=c["series"][2], label="with anti-windup")
        ax[2].axhline(0.95, lw=0.9, ls=(0, (4, 3)), color=c["axis"])
        ax[2].set_ylabel("duty")
        ax[2].set_xlabel("time (ms)")
        ax[2].set_ylim(0, 1.08)
        ax[2].set_title("Duty ratio; the dashed line is the modulator's limit",
                        loc="left")

        for a_ in ax:
            a_.axvspan(4.0, 10.0, color=c["muted"], alpha=0.10, lw=0)
            tidy(a_, c)

        fig.suptitle("Integrator windup during a line sag",
                     y=0.998, fontsize=11, color=c["ink"], ha="left", x=0.005)
        fig.tight_layout()

        o_on = 100 * (float(r_on["vout"][r_on["t"] >= 10e-3].max()) - 12) / 12
        o_off = 100 * (float(r_off["vout"][r_off["t"] >= 10e-3].max()) - 12) / 12
        footnote(fig,
                 f"Overshoot when the input recovers: {o_on:.0f} % with "
                 f"anti-windup, {o_off:.0f} % without. The difference is "
                 f"entirely what the integrator stored during the six "
                 f"milliseconds its output was pinned to the limit, and it "
                 f"gets worse the deeper the sag: a sag to 12 V costs "
                 f"47 %, to 11 V costs 75 %, to 10 V costs {o_off:.0f} %.", c)
        return fig

    render(build, os.path.join(OUT, "antiwindup"))


def fig_limit_cycle(designs):
    """Why DPWM resolution has to beat ADC resolution."""
    d = designs[-1]
    runs = {}
    for bits in (7, 12):
        p = BuckParams(n_dpwm=bits)
        runs[bits] = simulate(p, d["b"], d["a"], t_end=2.0e-3, dt=2e-8,
                              t_ramp=0.0, settled=True, averaged=False)

    def build(c):
        fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
        lsb = P.adc_range / 2 ** P.adc_bits

        for i, bits in enumerate((7, 12)):
            r = runs[bits]
            m = r["t"] >= 1.0e-3
            step_mv = 1e3 * P.vin / 2 ** bits
            ax[i].plot((r["t"][m] - 1.0e-3) * 1e3,
                       (r["vout"][m] - P.vref) * 1e3, lw=1.4,
                       color=c["series"][1 if bits == 7 else 2])
            ax[i].axhline(0, lw=0.9, ls=(0, (4, 3)), color=c["axis"])
            ax[i].set_ylabel("$V_{out} - 12$ V  (mV)")
            pp = float(np.max(r["vout"][m]) - np.min(r["vout"][m])) * 1e3
            ax[i].set_title(
                f"{bits}-bit DPWM: one duty step moves the output "
                f"{step_mv:.1f} mV, one ADC step is {lsb*1e3:.1f} mV  "
                f"->  {pp:.0f} mV pk-pk", loc="left")
            tidy(ax[i], c)

        ax[1].set_xlabel("time (ms)")
        fig.suptitle("Limit cycling, and the resolution rule that prevents it",
                     y=0.998, fontsize=11, color=c["ink"], ha="left", x=0.005)
        fig.tight_layout()
        footnote(fig,
                 "When one duty step moves the output by more than one ADC "
                 "step, no available duty puts the output on the target code. "
                 "The loop then alternates between the two codes on either "
                 "side of it forever. More loop gain cannot fix this, because "
                 "nothing is wrong with the gain; the cure is a modulator "
                 "finer than the converter's measurement of itself.", c)
        return fig

    render(build, os.path.join(OUT, "limit_cycle"))


# ----------------------------------------------------------------------
def main():
    print(f"plant: f_LC = {P.f_lc:.0f} Hz  Q = {P.q_lc:.1f}  "
          f"f_ESR = {P.f_esr/1e3:.1f} kHz  Ts = {P.ts*1e6:.0f} us")
    designs = build_designs()
    for d in designs:
        print(f"  {d['label']:16s} fz={d['rec']['f_zero']:6.0f} Hz  "
              f"fp={d['rec']['f_pole']:7.0f} Hz  PM={d['margins']['pm']:6.1f}  "
              f"GM={d['margins']['gm_db']:5.1f} dB  max|z|="
              f"{d['cl']['spectral_radius']:.4f}  "
              f"{'stable' if d['cl']['stable'] else 'UNSTABLE'}")

    print("drawing")
    fig_plant(P)
    fig_loop(designs)
    fig_poles(designs)
    fig_transient(designs)
    fig_ripple(designs)
    fig_windup(designs)
    fig_limit_cycle(designs)
    print("done")


if __name__ == "__main__":
    main()
