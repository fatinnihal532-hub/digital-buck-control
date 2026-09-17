"""
buck_blocks.py - a synchronous buck converter and the digital loop around it.

Two plants live here and they must agree, because the whole method depends on
it. The averaged plant is smooth and linear, and it is the one a compensator is
designed against on a Bode plot. The switching plant chops its input at the
switching frequency, and it is the one that tells you whether the design
survives contact with reality. If a controller designed on the first does not
work on the second, the model was wrong, not the controller.

Circuit, all quantities referred to ground:

      Vin --[ Q1 ]--+--[ rL  L ]--+---------+------ Vout
                    |             |         |
                  [ Q2 ]         [rc]      [ R ]
                    |             |         |
                    |            [C]        |
      ------------- +-------------+---------+

rL is the inductor's winding resistance and rc the output capacitor's
equivalent series resistance. Neither is optional. rc alone decides most of
the output ripple in a real design, and it places a zero in the control loop
that the compensator has to be told about.
"""

from __future__ import annotations

import numpy as np

from blockset import Block


class BuckParams:
    """Every number the converter needs, in one place.

    The defaults describe a small 24 V to 12 V point-of-load rail: 100 kHz
    switching, 2 A nominal, and passives that a real board would use.
    """

    def __init__(self, vin=24.0, vref=12.0, L=100e-6, C=100e-6,
                 rL=30e-3, rc=20e-3, R=6.0, fsw=100e3, n_dpwm=10,
                 adc_bits=12, adc_range=25.0):
        self.vin, self.vref = vin, vref
        self.L, self.C, self.rL, self.rc, self.R = L, C, rL, rc, R
        self.fsw, self.n_dpwm = fsw, n_dpwm
        self.adc_bits, self.adc_range = adc_bits, adc_range

    @property
    def ts(self):
        """Control period. Sampling once per switching cycle is what almost
        every digital supply does, because sampling at the same instant every
        cycle is what makes the switching ripple invisible to the loop."""
        return 1.0 / self.fsw

    @property
    def f_lc(self):
        """LC corner. The output filter contributes its full 180 degrees of
        phase lag above this, which is the single fact that decides what kind
        of compensator the converter needs."""
        return 1.0 / (2 * np.pi * np.sqrt(self.L * self.C))

    @property
    def f_esr(self):
        """The zero the output capacitor's ESR puts in the control path. A
        ceramic output cap pushes this so high that it stops being useful,
        which is why an all-ceramic design is harder to compensate, not
        easier."""
        return 1.0 / (2 * np.pi * self.rc * self.C)

    @property
    def q_lc(self):
        """Quality factor of the output filter at the nominal load."""
        return self.R * np.sqrt(self.C / self.L)

    def ripple_current(self, duty=None):
        """Peak-to-peak inductor ripple, from volt-seconds across L."""
        d = self.vref / self.vin if duty is None else duty
        return (self.vin - self.vref) * d / (self.L * self.fsw)

    def state_space(self):
        """Averaged small-signal model, states [iL, vC], input duty.

        Derived from the two circuit equations rather than quoted from a
        table, so that changing rc or rL changes the answer:

            vo  = vC + rc (iL - vo/R)      =>  vo = a (vC + rc iL),  a = R/(R+rc)
            L diL/dt = d Vin - rL iL - vo
            C dvC/dt = iL - vo/R
        """
        a = self.R / (self.R + self.rc)
        A = np.array([
            [-(self.rL + a * self.rc) / self.L, -a / self.L],
            [(1.0 - a * self.rc / self.R) / self.C, -a / (self.R * self.C)],
        ])
        B = np.array([[self.vin / self.L], [0.0]])
        C = np.array([[a * self.rc, a]])
        D = np.array([[0.0]])
        return A, B, C, D


class BuckSwitching(Block):
    """The converter itself, driven by a gate signal rather than a duty ratio.

    Synchronous rectification is assumed, so the low-side device conducts
    whenever the high-side one does not and the inductor current is free to go
    negative. That keeps the converter in continuous conduction at any load,
    which is what a synchronous buck really does and what makes the averaged
    model valid all the way down to no load.

    States are [iL, vC]; the output is [vout, iL, iout].
    """

    direct_feedthrough = False
    n_states = 2

    def __init__(self, name, p: BuckParams, x0=(0.0, 0.0)):
        super().__init__(name)
        self.p = p
        self.x0 = np.asarray(x0, dtype=float)

    def _vout(self, x, R):
        a = R / (R + self.p.rc)
        return a * (x[1] + self.p.rc * x[0])

    def out(self, t, x, u):
        R = self._load(t, u)
        vo = self._vout(x, R)
        return np.array([vo, x[0], vo / R])

    def _load(self, t, u):
        # input port 1 carries the load resistance, so a load step is just a
        # signal like any other rather than a special case inside the plant
        if len(u) > 1 and u[1] is not None:
            return float(u[1])
        return self.p.R

    def _vin(self, u):
        """Port 2 carries the input voltage.

        Making the supply a signal rather than a constant is what lets a line
        sag be simulated at all, and a sag is the scenario where a converter
        genuinely runs out of duty: if the input falls far enough that the
        required ratio exceeds the modulator's limit, the loop saturates and
        stays there until the input recovers.
        """
        if len(u) > 2 and u[2] is not None:
            return float(u[2])
        return self.p.vin

    def deriv(self, t, x, u):
        p = self.p
        s = float(u[0])                       # gate signal, 1 = high side on
        R = self._load(t, u)
        vo = self._vout(x, R)
        diL = (s * self._vin(u) - p.rL * x[0] - vo) / p.L
        dvC = (x[0] - vo / R) / p.C
        return np.array([diL, dvC])


class BuckAveraged(Block):
    """The same converter with the switching averaged away.

    Identical equations with the gate signal replaced by the duty ratio. It
    runs far faster and it is the model the compensator is designed against.
    Comparing its output against the switching model is the cheapest way to
    catch a modelling mistake.
    """

    direct_feedthrough = False
    n_states = 2

    def __init__(self, name, p: BuckParams, x0=(0.0, 0.0)):
        super().__init__(name)
        self.p = p
        self.x0 = np.asarray(x0, dtype=float)

    def out(self, t, x, u):
        p = self.p
        R = float(u[1]) if len(u) > 1 and u[1] is not None else p.R
        a = R / (R + p.rc)
        vo = a * (x[1] + p.rc * x[0])
        return np.array([vo, x[0], vo / R])

    def deriv(self, t, x, u):
        p = self.p
        d = float(np.clip(u[0], 0.0, 1.0))
        R = float(u[1]) if len(u) > 1 and u[1] is not None else p.R
        vin = float(u[2]) if len(u) > 2 and u[2] is not None else p.vin
        a = R / (R + p.rc)
        vo = a * (x[1] + p.rc * x[0])
        return np.array([(d * vin - p.rL * x[0] - vo) / p.L,
                         (x[0] - vo / R) / p.C])


class ADC(Block):
    """Sample, hold and quantise - what the loop actually measures.

    Quantisation is the reason a digital supply has a noise floor on its output
    voltage that no amount of loop gain removes. If the DPWM step is coarser
    than the ADC step, the duty cannot be adjusted finely enough to sit on the
    target code and the output hunts between two levels. That is limit-cycle
    oscillation, and the cure is either a finer DPWM or a deliberately coarser
    ADC, not more gain.
    """

    direct_feedthrough = False

    def __init__(self, name, p: BuckParams, channel=0):
        self.n_states = 1
        super().__init__(name)
        self.sample_time = p.ts
        self.p = p
        self.channel = channel
        self.lsb = p.adc_range / (2 ** p.adc_bits)
        self.x0 = np.array([0.0])

    def out(self, t, x, u):
        return x[0]

    def update(self, t, x, u):
        raw = np.asarray(u[0], dtype=float)
        v = float(raw[self.channel] if raw.ndim else raw)
        code = np.clip(np.floor(v / self.lsb), 0, 2 ** self.p.adc_bits - 1)
        return np.array([code * self.lsb])


class DiscreteCompensator(Block):
    """A digital compensator as a difference equation, with one sample of
    computation delay and clamping anti-windup.

    The delay is not an approximation added for realism: a real controller
    samples at the start of a cycle and the new duty only takes effect at the
    next PWM reload, so the loop genuinely carries a delay of about one and a
    half sample periods once the modulator's own half-period is counted. At a
    crossover of a tenth of the switching frequency that is 54 degrees of phase
    lag, which is more than most compensators have to give. Designing without
    it produces a loop that looks fine on paper and rings on the bench.

    Anti-windup is by clamping: while the output sits on a limit, the states
    are not advanced. Without it, a long saturation during start-up charges
    the integrator far past what is needed, and the output overshoots by
    whatever was stored there.
    """

    direct_feedthrough = False

    def __init__(self, name, b, a, ts, lo=0.0, hi=1.0, y0=None,
                 antiwindup=True):
        self.b = np.asarray(b, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.n = max(len(self.b), len(self.a))
        # state layout: [u history (n), y history (n-1), held output]
        self.n_states = self.n + (self.n - 1) + 1
        super().__init__(name)
        self.sample_time = float(ts)
        self.lo, self.hi = lo, hi
        self.antiwindup = bool(antiwindup)
        self.x0 = np.zeros(self.n_states)

        if y0 is not None:
            # Start the compensator already holding a given output, so a run
            # can begin in steady state instead of spending milliseconds
            # getting there. Zero input with a constant output history is a
            # genuine fixed point of the difference equation whenever the
            # compensator contains an integrator: the denominator then has a
            # root at z = 1, which means its coefficients sum to zero, and the
            # recursion reproduces the same output forever. Asserting that
            # rather than assuming it catches a compensator that has no
            # integrator, where this trick would silently inject a transient.
            if abs(float(np.sum(self.a))) > 1e-9:
                raise ValueError("y0 needs a compensator with an integrator "
                                 "(denominator coefficients must sum to zero)")
            self.x0[self.n:] = float(y0)

    def out(self, t, x, u):
        return x[-1]

    def update(self, t, x, u):
        n = self.n
        uh = x[:n].copy()
        yh = x[n:n + n - 1].copy()

        uh = np.roll(uh, 1)
        uh[0] = float(u[0])

        acc = float(np.dot(self.b, uh[:len(self.b)]))
        if len(self.a) > 1:
            acc -= float(np.dot(self.a[1:], yh[:len(self.a) - 1]))
        y_raw = acc / self.a[0]
        y = float(np.clip(y_raw, self.lo, self.hi))

        yh = np.roll(yh, 1)
        if len(yh):
            # The whole of anti-windup is this one choice: which value goes
            # into the output history.
            #
            # With it, the recursion is fed the duty that was actually
            # applied, so the compensator's memory matches what the converter
            # really did. Without it, the recursion is fed the duty the
            # compensator asked for, and while the output sits on a limit the
            # difference between the two accumulates in the states. When the
            # error finally changes sign, all of that stored surplus has to be
            # unwound before the output can come off the limit, and the
            # converter sails past the target in the meantime.
            #
            # Note that the input history advances either way. Freezing it too
            # would not be anti-windup, it would be deafness: the compensator
            # would stop hearing the error while saturated.
            yh[0] = y if self.antiwindup else y_raw
        return np.concatenate([uh, yh, [y]])


class DPWM(Block):
    """Digital pulse-width modulator: quantise the duty, then compare.

    The duty a digital controller can ask for is not continuous. A counter of
    N bits running from a clock gives 2^N settings, and the output voltage
    resolution that follows is Vin / 2^N. At 24 V and ten bits that is 23 mV
    per step, which has to be finer than one ADC step or the loop limit-cycles.
    """

    def __init__(self, name, p: BuckParams):
        super().__init__(name)
        self.p = p
        self.levels = 2 ** p.n_dpwm

    def out(self, t, x, u):
        d = float(np.clip(u[0], 0.0, 1.0))
        d_q = np.round(d * self.levels) / self.levels
        ramp = (t * self.p.fsw) % 1.0          # trailing-edge modulation
        return 1.0 if ramp < d_q else 0.0


class Profile(Block):
    """A piecewise-constant signal from a list of (time, value) pairs.

    Used for the load, so a run can include an overload and its removal rather
    than a single step. Recovery from an overload is where a compensator's
    anti-windup earns its place, and a one-step profile never reaches it.
    """

    def __init__(self, name, points):
        super().__init__(name)
        self.points = sorted(points, key=lambda kv: kv[0])

    def out(self, t, x, u):
        value = self.points[0][1]
        for tk, vk in self.points:
            if t >= tk:
                value = vk
            else:
                break
        return value


class SoftStart(Block):
    """Ramps the reference instead of applying it as a step.

    A step reference asks the loop for an infinite slew rate, and the only way
    the converter can answer is by saturating the duty. Ramping the reference
    over a few milliseconds keeps the loop in its linear region the whole way
    up, which limits the inrush current and stops the output overshooting on
    arrival.
    """

    def __init__(self, name, vref, t_ramp):
        super().__init__(name)
        self.vref, self.t_ramp = vref, t_ramp

    def out(self, t, x, u):
        if self.t_ramp <= 0:
            return self.vref
        return self.vref * min(1.0, t / self.t_ramp)
