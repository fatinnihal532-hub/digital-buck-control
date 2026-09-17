"""
buck_model.py - the closed loop, wired up.

    soft-start ref ---->(+)
                         |  e        compensator      1/z        DPWM
                        (-)--------> Gc(z) --------> delay ----> gate
                         ^                                        |
                         |                                        v
                        ADC <-------- vout <------------------ converter
                     (sample,                                     ^
                      hold,                                       |
                      quantise)                          load step signal

Everything the loop cannot do instantly is a block in that diagram rather than
an assumption in a comment. The ADC quantises. The delay is a real 1/z. The
modulator holds the duty for a full period and can only take 2^N values. Those
four effects are what separate a digital supply from the continuous textbook
one, and leaving any of them out makes the simulation agree with the maths and
disagree with the bench.
"""

from __future__ import annotations

import numpy as np

from blockset import Model, Scope, Step, Sum, UnitDelay
from models.buck_blocks import (ADC, DPWM, BuckAveraged, BuckParams,
                                BuckSwitching, DiscreteCompensator, Profile,
                                SoftStart)


def build(p: BuckParams, b, a, *, dt=2e-8, t_ramp=2e-3,
          load_step=None, averaged=False, settled=False,
          antiwindup=True, vin_profile=None, name="buck"):
    """Assemble the closed loop.

    load_step is (time, new resistance) or None. dt defaults to 20 ns, which
    puts 500 solver steps inside a 10 us switching period - enough that the
    duty the modulator actually produces is within 0.2 percent of the one it
    was asked for.
    """
    mdl = Model(name, dt=dt)

    ref = mdl.add(SoftStart("ref", vref=p.vref, t_ramp=t_ramp))
    if load_step is None:
        load = mdl.add(Step("load", 0.0, p.R, p.R))
    elif isinstance(load_step[0], (tuple, list)):
        load = mdl.add(Profile("load", load_step))     # (t, R) pairs
    else:
        t_step, r_new = load_step
        load = mdl.add(Step("load", t_step, p.R, r_new))

    # Starting settled costs one line here and saves a great deal of run time
    # when the quantity of interest is steady-state ripple rather than the
    # start-up transient: the switching model is expensive, and there is no
    # point integrating 20 ns steps through a millisecond of settling that the
    # averaged model has already shown.
    # The seed has to include the winding-resistance drop, or the run starts
    # with a small duty deficit and spends its first quarter-millisecond
    # correcting it - which then shows up in a ripple measurement as a slope
    # that is not ripple at all.
    #   vout = D*vin - iL*rL   =>   D = (vref + rL*vref/R) / vin
    duty0 = (p.vref + p.rL * p.vref / p.R) / p.vin
    x0_plant = (p.vref / p.R, p.vref) if settled else (0.0, 0.0)

    plant_cls = BuckAveraged if averaged else BuckSwitching
    plant = mdl.add(plant_cls("plant", p, x0=x0_plant))

    adc = mdl.add(ADC("adc", p, channel=0))
    err = mdl.add(Sum("err", "+-"))
    comp = mdl.add(DiscreteCompensator("comp", b, a, ts=p.ts, lo=0.0, hi=0.95,
                                       y0=duty0 if settled else None,
                                       antiwindup=antiwindup))
    delay = mdl.add(UnitDelay("delay", sample_time=p.ts,
                              x0=duty0 if settled else 0.0))
    # The modulator only exists in the switching model. Leaving an unwired
    # DPWM block in the averaged diagram would still have it evaluated every
    # step, which is how a block with nothing on its input port ends up being
    # asked for an output it cannot produce.
    traces = ["ref", "plant", "adc", "comp", "delay", "load", "err"]
    dpwm = None
    if not averaged:
        dpwm = mdl.add(DPWM("dpwm", p))
        traces.append("dpwm")
    scope = mdl.add(Scope("scope", traces))

    mdl.connect(plant, adc)
    mdl.connect(ref, err, port=0)
    mdl.connect(adc, err, port=1)
    mdl.connect(err, comp)
    mdl.connect(comp, delay)
    if averaged:
        mdl.connect(delay, plant, port=0)      # duty straight in
    else:
        mdl.connect(delay, dpwm)
        mdl.connect(dpwm, plant, port=0)       # gate signal
    mdl.connect(load, plant, port=1)

    vin_blk = mdl.add(Profile("vin", vin_profile or [(0.0, p.vin)]))
    mdl.connect(vin_blk, plant, port=2)
    scope.signals.append("vin")
    scope.data["vin"] = []
    return mdl, scope


def simulate(p: BuckParams, b, a, t_end, **kw):
    mdl, scope = build(p, b, a, **kw)
    mdl.run(t_end)

    t = scope.time()
    plant = np.array(scope.data["plant"])
    return dict(
        t=t,
        vout=plant[:, 0],
        il=plant[:, 1],
        iout=plant[:, 2],
        ref=np.array(scope.data["ref"], dtype=float),
        duty=np.array(scope.data["delay"], dtype=float),
        gate=(np.array(scope.data["dpwm"], dtype=float)
              if "dpwm" in scope.data else None),
        load=np.array(scope.data["load"], dtype=float),
        vin=np.array(scope.data["vin"], dtype=float),
        err=np.array(scope.data["err"], dtype=float),
        params=p,
    )


# ----------------------------------------------------------------------
# measurements taken from a run, defined the way a datasheet defines them
# ----------------------------------------------------------------------

def ripple_pp(t, y, t_from, t_to):
    """Peak-to-peak of a signal over a settled window."""
    m = (t >= t_from) & (t <= t_to)
    seg = y[m]
    return float(seg.max() - seg.min())


def load_step_metrics(t, vout, t_step, vref, settle_band=0.01, window=2e-3):
    """Undershoot and recovery time after a load step.

    Recovery time is measured to the point after which the output stays inside
    the band for good, not the first time it happens to touch it. The
    difference matters on an underdamped loop, where the output crosses the
    band several times on the way in.
    """
    m = (t >= t_step) & (t <= t_step + window)
    tt, vv = t[m], vout[m]
    if len(vv) == 0:
        return dict(deviation=np.nan, recovery_s=np.nan)

    dev = float(np.max(np.abs(vv - vref)))
    band = settle_band * vref
    outside = np.where(np.abs(vv - vref) > band)[0]
    rec = float(tt[outside[-1]] - t_step) if len(outside) else 0.0
    return dict(deviation=dev, deviation_pct=100.0 * dev / vref,
                recovery_s=rec, band_pct=100.0 * settle_band)


def startup_metrics(t, vout, vref, settle_band=0.01):
    """Overshoot on arrival and the time taken to get there."""
    peak = float(np.max(vout))
    over = 100.0 * (peak - vref) / vref
    band = settle_band * vref
    outside = np.where(np.abs(vout - vref) > band)[0]
    settle = float(t[outside[-1]]) if len(outside) else 0.0
    return dict(peak=peak, overshoot_pct=over, settle_s=settle)
