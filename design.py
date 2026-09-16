"""
design.py - designing the compensator on paper, before any simulation runs.

This is the half of the project that a simulation cannot replace. A loop that
happens to look stable in one time-domain run has not been shown to be stable;
a loop with a stated crossover frequency and phase margin has. Everything here
works on the averaged model from buck_blocks.py, and the switching simulation
afterwards exists to check that the paper answer survives.

Method: the k-factor. Pick a crossover frequency and a phase margin, ask the
plant what phase it has there, and give the compensator exactly the phase boost
that makes up the difference. The zeros and poles are then placed symmetrically
in log frequency around the crossover, which is what "k" is.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from models.buck_blocks import BuckParams


def plant_response(p: BuckParams, f):
    """Control-to-output response Gvd(jw), including the loop's delay.

    Two delays are folded in, and both are real:

      one full sample  the controller samples at the top of a cycle and the
                       result cannot reach the modulator until the next reload
      half a sample    the modulator itself holds the duty for a cycle, which
                       on average delays the response by half a period

    Together that is 1.5 sample periods. It contributes no attenuation at all,
    only phase lag that grows linearly with frequency, which is why it is
    invisible on a gain plot and lethal on a phase plot.
    """
    A, B, C, D = p.state_space()
    w = 2 * np.pi * np.asarray(f, dtype=float)
    _, H = signal.freqresp(signal.StateSpace(A, B, C, D), w=w)
    delay = np.exp(-1j * w * 1.5 * p.ts)
    return H * delay


def plant_phase_at(p: BuckParams, f_cross, n=4000):
    """Plant phase at one frequency, unwrapped from DC upward.

    A single call to np.angle cannot give this. It returns a principal value in
    (-180, 180], so a plant that has genuinely fallen to -215 degrees is
    reported as +145, and a compensator designed from that number is designed
    for the wrong plant by a full turn. The phase has to be followed
    continuously from low frequency to the crossover, which is what the sweep
    and the unwrap below do.
    """
    f = np.logspace(np.log10(f_cross) - 4, np.log10(f_cross), n)
    ph = np.unwrap(np.angle(plant_response(p, f)))
    return float(np.degrees(ph[-1])), float(abs(plant_response(p, np.array([f_cross]))[0]))


def _k_from_phase(phi_needed_deg, order):
    """Solve for the k-factor that supplies a required phase at crossover.

    For a type II compensator the phase at crossover is 2*atan(k) - 180
    degrees; for type III it is 4*atan(k) - 270. Both follow from the fact
    that a zero at wc/k and a pole at wc*k contribute equal and opposite
    angles about the crossover. Inverting them is exact, so no search is
    needed.
    """
    if order == 2:
        theta = np.deg2rad((phi_needed_deg + 180.0) / 2.0)
    elif order == 3:
        theta = np.deg2rad((phi_needed_deg + 270.0) / 4.0)
    else:
        raise ValueError("order must be 2 or 3")
    if not (0.0 < theta < np.pi / 2):
        raise ValueError(
            f"a type {order} compensator cannot supply {phi_needed_deg:.1f} "
            f"degrees at crossover - move the crossover or accept less margin")
    return float(np.tan(theta))


def design(p: BuckParams, f_cross, pm_deg, order=3):
    """Return the continuous compensator (num, den) and its design record."""
    phi_p, mag_p = plant_phase_at(p, f_cross)

    phi_needed = -180.0 + pm_deg - phi_p          # what Gc must have at wc
    k = _k_from_phase(phi_needed, order)

    wc = 2 * np.pi * f_cross
    wz, wp = wc / k, wc * k

    if order == 2:
        kc = wc / (k * mag_p)
        num = np.polymul([kc], [1.0 / wz, 1.0])
        den = np.polymul([1.0, 0.0], [1.0 / wp, 1.0])
    else:
        kc = wc / (k * k * mag_p)
        z = np.polymul([1.0 / wz, 1.0], [1.0 / wz, 1.0])
        num = np.polymul([kc], z)
        den = np.polymul([1.0, 0.0],
                         np.polymul([1.0 / wp, 1.0], [1.0 / wp, 1.0]))

    record = dict(order=order, f_cross=f_cross, pm_target=pm_deg,
                  f_zero=wz / (2 * np.pi), f_pole=wp / (2 * np.pi),
                  k=k, kc=kc, plant_phase_deg=phi_p, plant_gain_db=20*np.log10(mag_p))
    return (np.asarray(num, dtype=float), np.asarray(den, dtype=float)), record


def to_discrete(num, den, ts, f_prewarp):
    """Tustin with pre-warping, so the compensator's corners land where they
    were designed.

    Plain Tustin compresses the whole frequency axis as it folds it into the
    unit circle, and the error grows towards half the sample rate. Pre-warping
    at the crossover forces the mapping to be exact at the one frequency the
    phase margin is measured at, and lets the error fall where it does not
    matter.
    """
    w0 = 2 * np.pi * f_prewarp
    alpha = w0 / np.tan(w0 * ts / 2.0)
    b, a = signal.bilinear(num, den, fs=alpha / 2.0)
    return np.asarray(b, dtype=float), np.asarray(a, dtype=float)


def loop_response(p: BuckParams, num, den, f):
    """Open-loop T(jw) = Gc * Gp, the quantity margins are read from."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    _, Gc = signal.freqresp(signal.TransferFunction(num, den), w=w)
    return Gc * plant_response(p, f)


def closed_loop_poles(p: BuckParams, b, a):
    """Closed-loop poles of the sampled system, which is the real stability test.

    Phase margin is a shortcut, and on this plant the shortcut breaks. A
    lightly damped output filter can push the loop gain back above unity at
    its resonance, so the magnitude crosses 0 dB three times. Phase margin is
    defined at one crossing and says nothing about the other two, and a loop
    can show a negative margin at the highest crossing while remaining
    perfectly stable - or the reverse.

    What settles it is where the closed-loop poles sit. Every pole inside the
    unit circle means stable, and the largest magnitude says how much damping
    is left. The plant is discretised with a zero-order hold because that is
    what the modulator physically does, and the controller's computation delay
    enters as an honest z^-1 rather than as a phase correction.
    """
    A, B, C, D = p.state_space()
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), p.ts, method="zoh")
    num_p, den_p = signal.ss2tf(Ad, Bd, Cd, Dd)
    num_p = np.atleast_1d(np.squeeze(num_p))

    # one sample of computation delay: multiply by z^-1, i.e. shift the
    # denominator up by one order
    num_ol = np.polymul(np.asarray(b, dtype=float), num_p)
    den_ol = np.polymul(np.asarray(a, dtype=float),
                        np.polymul(den_p, [1.0, 0.0]))

    n = max(len(num_ol), len(den_ol))
    num_ol = np.pad(num_ol, (n - len(num_ol), 0))
    den_ol = np.pad(den_ol, (n - len(den_ol), 0))

    poles = np.roots(den_ol + num_ol)
    rho = float(np.max(np.abs(poles))) if len(poles) else 0.0
    return dict(poles=poles, spectral_radius=rho, stable=rho < 1.0)


def margins(p: BuckParams, num, den):
    """Gain crossover, phase margin and gain margin, found by interpolation.

    Read off the same curve an interviewer would ask you to sketch: crossover
    is where the magnitude passes unity, phase margin is how far the phase is
    from -180 there, and gain margin is how much gain could be added before
    the phase reaches -180.
    """
    f = np.logspace(0, np.log10(0.5 / p.ts), 40000)
    T = loop_response(p, num, den, f)
    mag = np.abs(T)
    ph = np.unwrap(np.angle(T))

    # The last crossing, not the first. A lightly damped output filter puts a
    # resonant peak in the loop gain, so the magnitude can dip below unity,
    # climb back through it at the resonance and fall through a third time.
    # The gain crossover that the phase margin refers to is the highest-
    # frequency one; taking the first gives a margin belonging to a crossing
    # the loop passes straight through.
    idx = np.where(np.diff(np.sign(mag - 1.0)))[0]
    if len(idx) == 0:
        return dict(f_cross=np.nan, pm=np.nan, gm_db=np.nan, stable=False)
    i = idx[-1]
    n_crossings = len(idx)
    fc = float(np.interp(0.0, [np.log10(mag[i + 1]), np.log10(mag[i])],
                         [f[i + 1], f[i]]))
    phc = float(np.interp(np.log10(fc), np.log10(f), ph))
    pm = float(np.degrees(phc) + 180.0)

    target = -np.pi
    j = np.where(np.diff(np.sign(ph - target)))[0]
    if len(j):
        jj = j[0]
        f180 = float(np.interp(target, [ph[jj], ph[jj + 1]], [f[jj], f[jj + 1]]))
        m180 = float(np.interp(np.log10(f180), np.log10(f), np.log10(mag)))
        gm_db = float(-20.0 * m180)
    else:
        gm_db = float("inf")

    return dict(f_cross=fc, pm=pm, gm_db=gm_db, stable=pm > 0,
                n_crossings=n_crossings,
                f=f, mag=mag, phase_deg=np.degrees(ph))
