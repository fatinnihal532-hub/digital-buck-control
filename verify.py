#!/usr/bin/env python3
"""
verify.py - checks the converter and the loop against closed-form theory.

Three kinds of check live here, and the third is the one that matters most.

Against theory       ripple, corner frequencies, steady-state duty: quantities
                     with a one-line derivation that the model must reproduce.
Against the design   the compensator was designed for a stated crossover and
                     phase margin, so the finished loop must actually have
                     them. A design method that does not hit its own target is
                     not a design method.
Against each other   the averaged model and the switching model describe the
                     same converter. If they disagree, one of them is wrong,
                     and no amount of agreement with theory in isolation would
                     have caught it.

Run it with:  python3 verify.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from models.buck_blocks import BuckParams
from models.buck_model import simulate
from models.design import (closed_loop_poles, design, margins, to_discrete)

PASSED = FAILED = 0


def check(name, measured, expected, tol_pct, note=""):
    global PASSED, FAILED
    err = 100.0 * (measured - expected) / expected if expected else measured
    ok = abs(err) <= tol_pct
    PASSED += ok
    FAILED += not ok
    print(f"  [{'pass' if ok else 'FAIL'}] {name}")
    print(f"         model {measured:12.5f}   theory {expected:12.5f}   "
          f"error {err:+7.3f} %   tolerance {tol_pct:.2f} %")
    if note:
        print(f"         {note}")


def check_true(name, ok, detail=""):
    global PASSED, FAILED
    PASSED += bool(ok)
    FAILED += not ok
    print(f"  [{'pass' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")


def main():
    P = BuckParams(n_dpwm=12)
    print("Digitally controlled buck converter - model against theory")
    print("=" * 70)
    print(f"  24 V to 12 V, {P.fsw/1e3:.0f} kHz, L = {P.L*1e6:.0f} uH, "
          f"C = {P.C*1e6:.0f} uF, load {P.R:.0f} ohm")

    # ---------------------------------------------------------------
    print("\n1. The output filter's own corner frequencies")
    check("LC corner", P.f_lc, 1 / (2 * np.pi * np.sqrt(P.L * P.C)), 0.01)
    check("ESR zero", P.f_esr, 1 / (2 * np.pi * P.rc * P.C), 0.01,
          note="at 80 kHz this zero is too high to help the compensator, "
               "which is exactly why a type III is needed")

    # ---------------------------------------------------------------
    print("\n2. Steady state in the switching model")
    print("   Ripple follows from volt-seconds across the inductor, and the")
    print("   duty from the conversion ratio corrected for the winding drop.")
    (num, den), _ = design(P, f_cross=4000, pm_deg=50, order=3)
    b, a = to_discrete(num, den, P.ts, f_prewarp=4000)
    rs = simulate(P, b, a, t_end=500e-6, dt=2e-8, t_ramp=0.0,
                  settled=True, averaged=False)
    w = rs["t"] > 400e-6
    il_pp = float(np.max(rs["il"][w]) - np.min(rs["il"][w]))
    vo_pp = float(np.max(rs["vout"][w]) - np.min(rs["vout"][w]))

    check("inductor ripple, pk-pk", il_pp, P.ripple_current(), 5.0,
          note="(Vin - Vout) * D / (L * fsw); the model also carries rL, "
               "which the expression does not")
    check("mean output voltage", float(rs["vout"][w].mean()), P.vref, 0.1)
    check("mean inductor current", float(rs["il"][w].mean()),
          P.vref / P.R, 1.0)

    esr = P.rc * P.ripple_current()
    cap = P.ripple_current() / (8 * P.C * P.fsw)
    check_true("output ripple below the ESR-plus-capacitor bound",
               vo_pp <= (esr + cap) * 1.02,
               f"measured {vo_pp*1e3:.2f} mV against a bound of "
               f"{(esr+cap)*1e3:.2f} mV ({esr*1e3:.1f} mV of it from ESR). "
               f"The two components peak at different instants, so their sum "
               f"is an upper bound and not an equality")

    # ---------------------------------------------------------------
    print("\n3. The compensator hits the target it was designed for")
    print("   The k-factor placement is exact, so this is a check on the")
    print("   algebra rather than on judgement. If it drifts, the phase")
    print("   unwrapping or the delay model has broken.")
    for order, fc, pm in ((2, 200.0, 60.0), (3, 4000.0, 50.0),
                          (3, 2500.0, 55.0)):
        (n_, d_), rec = design(P, f_cross=fc, pm_deg=pm, order=order)
        m = margins(P, n_, d_)
        check(f"type {order}: crossover", m["f_cross"], fc, 1.0)
        check(f"type {order}: phase margin", m["pm"], pm, 1.0)

    # ---------------------------------------------------------------
    print("\n4. The compensator is implementable at this sample rate")
    print("   A pole placed above the Nyquist frequency cannot survive")
    print("   discretisation. Checking it here stops a design that only")
    print("   exists on the continuous-time page.")
    (n_, d_), rec = design(P, f_cross=4000, pm_deg=50, order=3)
    check_true("both compensator poles below Nyquist",
               rec["f_pole"] < 0.5 / P.ts,
               f"pole at {rec['f_pole']/1e3:.1f} kHz against a Nyquist "
               f"frequency of {0.5/P.ts/1e3:.0f} kHz")

    # ---------------------------------------------------------------
    print("\n5. Stability, decided by the closed-loop poles")
    print("   Phase margin is read at one gain crossover. This plant has a")
    print("   resonance that can produce three, so the margin can disagree")
    print("   with the truth. The poles cannot.")
    for label, order, fc, pm, expect in (
            ("type II at 200 Hz", 2, 200.0, 60.0, True),
            ("type II at 600 Hz", 2, 600.0, 60.0, False),
            ("type III at 4 kHz", 3, 4000.0, 50.0, True)):
        (n_, d_), _ = design(P, f_cross=fc, pm_deg=pm, order=order)
        bb, aa = to_discrete(n_, d_, P.ts, f_prewarp=fc)
        cl = closed_loop_poles(P, bb, aa)
        check_true(f"{label} is {'stable' if expect else 'unstable'}",
                   cl["stable"] == expect,
                   f"max |z| = {cl['spectral_radius']:.4f}")

    # ---------------------------------------------------------------
    print("\n6. The two plant models agree")
    print("   The averaged model is what the compensator was designed")
    print("   against; the switching model is what it has to survive. Running")
    print("   both through the same load step and comparing is the check that")
    print("   catches a modelling error neither would reveal alone.")
    kw = dict(t_end=3.2e-3, t_ramp=1e-3, load_step=(2.5e-3, 3.0))
    sw = simulate(P, b, a, dt=2e-8, averaged=False, **kw)
    av = simulate(P, b, a, dt=2e-7, averaged=True, **kw)

    ws = (sw["t"] >= 2.5e-3)
    wa = (av["t"] >= 2.5e-3)
    dip_sw = P.vref - float(sw["vout"][ws].min())
    dip_av = P.vref - float(av["vout"][wa].min())
    check("load-step undershoot, switching against averaged",
          dip_sw, dip_av, 2.0,
          note=f"{dip_sw*1e3:.0f} mV against {dip_av*1e3:.0f} mV; the "
               f"switching model carries ripple on top, so exact equality "
               f"is not expected")

    print("\n" + "=" * 70)
    print(f"{PASSED} checks passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
