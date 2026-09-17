"""
core.py - a small block-diagram simulation engine.

The point of this file is to make a Python model read the way a Simulink model
reads: you build a diagram out of blocks, you wire the blocks together, you
press run, and a scope collects the traces. Nothing here is specific to power
electronics.

Three ideas carry the whole engine, and they are the same three Simulink uses.

1. A block is a function of time, its own state and its inputs. It exposes an
   output equation y = g(t, x, u) and, if it has continuous state, a derivative
   equation dx/dt = f(t, x, u).

2. Execution order is worked out once, before the run, by a topological sort
   over the blocks that pass their input straight to their output. A block
   without direct feedthrough - an integrator, a unit delay - is what breaks a
   feedback loop, which is why Simulink complains about algebraic loops only
   when every block in the loop is direct feedthrough.

3. Continuous states are advanced by a fixed-step RK4 solver. Discrete blocks
   hold their output between their own sample instants, so a controller running
   at 50 kHz and a power stage integrated at 2 MHz coexist in one run. That is
   a hybrid simulation, and it is what makes a digitally controlled converter
   honest: the controller really does see a held, sampled version of the plant.
"""

from __future__ import annotations

import numpy as np


class Block:
    """Base class. Subclasses override out(), and deriv() or update()."""

    #: True when the output at time t depends on the input at time t.
    #: Set False for integrators and delays - these are the blocks that let a
    #: feedback loop close without becoming an algebraic loop.
    direct_feedthrough = True

    #: Number of continuous states. 0 means the block has none.
    n_states = 0

    #: None for a continuous block, otherwise the sample period in seconds.
    sample_time = None

    def __init__(self, name):
        self.name = name
        self.inputs = []          # list of (block, port) or (None, constant)
        self.x0 = np.zeros(self.n_states)

    # ---- the two equations a block may define -------------------------
    def out(self, t, x, u):
        """Output equation y = g(t, x, u)."""
        raise NotImplementedError

    def deriv(self, t, x, u):
        """Continuous derivative dx/dt = f(t, x, u)."""
        return np.zeros(self.n_states)

    def update(self, t, x, u):
        """Discrete state update, called at each sample instant."""
        return x

    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r}>"


class Model:
    """A diagram: blocks, the wires between them, and a fixed-step solver."""

    def __init__(self, name, dt):
        self.name = name
        self.dt = float(dt)       # the solver's fixed step
        self.blocks = []
        self._order = None

    # ---- building the diagram ----------------------------------------
    def add(self, block):
        self.blocks.append(block)
        self._order = None
        return block

    def connect(self, src, dst, port=0):
        """Wire src's output into dst's input port.

        src may be a Block or a plain number, which stands in for a constant
        source the way a Constant block would.
        """
        while len(dst.inputs) <= port:
            dst.inputs.append((None, 0.0))
        dst.inputs[port] = (src, None) if isinstance(src, Block) else (None, src)
        self._order = None
        return dst

    # ---- execution order ---------------------------------------------
    def _sort(self):
        """Topological sort over direct-feedthrough edges.

        An edge src -> dst is only a scheduling constraint when dst actually
        needs src's value at this instant. A block with direct_feedthrough
        False can be evaluated from its own state alone, so it contributes no
        incoming edges and is free to be scheduled first.
        """
        pending = {b: set() for b in self.blocks}
        for b in self.blocks:
            if not b.direct_feedthrough:
                continue
            for src, _ in b.inputs:
                if isinstance(src, Block):
                    pending[b].add(src)

        order, done = [], set()
        while len(order) < len(self.blocks):
            ready = [b for b in self.blocks
                     if b not in done and pending[b] <= done]
            if not ready:
                stuck = [b.name for b in self.blocks if b not in done]
                raise RuntimeError(
                    "algebraic loop: every block in this loop feeds its input "
                    "straight through to its output - " + ", ".join(stuck))
            for b in ready:
                order.append(b)
                done.add(b)
        self._order = order
        return order

    # ---- one sweep of the diagram ------------------------------------
    def _outputs(self, t, X, cont_slice, disc_state):
        """Evaluate every block's output at time t for continuous state X."""
        y = {}
        for b in self._order:
            # A block without direct feedthrough is evaluated before its own
            # sources, which is exactly what lets it break a feedback loop.
            # Its output cannot depend on u, so an input that has not been
            # computed yet is handed over as None rather than waited for.
            u = [y.get(src.name) if isinstance(src, Block) else const
                 for src, const in b.inputs]
            if b.sample_time is not None:
                xb = disc_state[b.name]
            elif b.n_states:
                xb = X[cont_slice[b.name]]
            else:
                xb = np.zeros(0)
            y[b.name] = b.out(t, xb, u)
        return y

    # ---- the run ------------------------------------------------------
    def run(self, t_end, progress=None):
        if self._order is None:
            self._sort()

        # lay every continuous state out in one vector so one RK4 call
        # advances the whole diagram together
        cont_slice, n = {}, 0
        for b in self.blocks:
            if b.sample_time is None and b.n_states:
                cont_slice[b.name] = slice(n, n + b.n_states)
                n += b.n_states
        X = np.zeros(n)
        for b in self.blocks:
            if b.name in cont_slice:
                X[cont_slice[b.name]] = b.x0

        disc_state = {b.name: np.array(b.x0, dtype=float)
                      for b in self.blocks if b.sample_time is not None}
        next_sample = {b.name: 0.0
                       for b in self.blocks if b.sample_time is not None}

        def f(t, Xv):
            """Global derivative. Discrete states are held, which is the
            zero-order hold that makes this a hybrid simulation."""
            y = self._outputs(t, Xv, cont_slice, disc_state)
            dX = np.zeros_like(Xv)
            for b in self.blocks:
                if b.name not in cont_slice:
                    continue
                u = [y[src.name] if isinstance(src, Block) else const
                     for src, const in b.inputs]
                dX[cont_slice[b.name]] = b.deriv(t, Xv[cont_slice[b.name]], u)
            return dX

        dt = self.dt
        steps = int(round(t_end / dt))
        for k in range(steps + 1):
            t = k * dt

            # discrete blocks fire first, on their own sample instants
            for b in self.blocks:
                if b.sample_time is None:
                    continue
                if t + 1e-12 >= next_sample[b.name]:
                    y = self._outputs(t, X, cont_slice, disc_state)
                    u = [y[src.name] if isinstance(src, Block) else const
                         for src, const in b.inputs]
                    disc_state[b.name] = b.update(
                        t, disc_state[b.name], u)
                    next_sample[b.name] += b.sample_time

            y = self._outputs(t, X, cont_slice, disc_state)
            for b in self.blocks:
                if isinstance(b, Scope):
                    b.log(t, y)

            if k == steps:
                break

            # classic fixed-step RK4 over the whole continuous state.
            # A diagram with no continuous state at all - a modulator feeding
            # a bridge, with the load left off - skips this entirely instead
            # of paying for four derivative evaluations that return nothing.
            if n:
                k1 = f(t, X)
                k2 = f(t + dt / 2, X + dt / 2 * k1)
                k3 = f(t + dt / 2, X + dt / 2 * k2)
                k4 = f(t + dt, X + dt * k3)
                X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            if progress and k % progress == 0:
                print(f"  t = {t:7.4f} s", flush=True)


# ======================================================================
# The blockset
# ======================================================================

class Constant(Block):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value

    def out(self, t, x, u):
        return self.value


class Gain(Block):
    def __init__(self, name, k):
        super().__init__(name)
        self.k = k

    def out(self, t, x, u):
        return self.k * u[0]


class Sum(Block):
    """Signs is a string of + and -, one per input port, as in Simulink."""

    def __init__(self, name, signs="+-"):
        super().__init__(name)
        self.signs = signs

    def out(self, t, x, u):
        total = 0.0
        for s, ui in zip(self.signs, u):
            total = total + (ui if s == "+" else -ui)
        return total


class Product(Block):
    def out(self, t, x, u):
        p = u[0]
        for ui in u[1:]:
            p = p * ui
        return p


class Saturation(Block):
    def __init__(self, name, lo, hi):
        super().__init__(name)
        self.lo, self.hi = lo, hi

    def out(self, t, x, u):
        return np.clip(u[0], self.lo, self.hi)


class Step(Block):
    def __init__(self, name, t_step, before, after):
        super().__init__(name)
        self.t_step, self.before, self.after = t_step, before, after

    def out(self, t, x, u):
        return self.after if t >= self.t_step else self.before


class Sine(Block):
    def __init__(self, name, amplitude, freq_hz, phase_rad=0.0, bias=0.0):
        super().__init__(name)
        self.a, self.f, self.ph, self.bias = amplitude, freq_hz, phase_rad, bias

    def out(self, t, x, u):
        return self.bias + self.a * np.sin(2 * np.pi * self.f * t + self.ph)


class Triangle(Block):
    """Unit triangular carrier, +/-1, the wave a PWM modulator compares to."""

    def __init__(self, name, freq_hz):
        super().__init__(name)
        self.f = freq_hz

    def out(self, t, x, u):
        frac = (t * self.f) % 1.0
        return 4.0 * abs(frac - 0.5) - 1.0


class Sawtooth(Block):
    """Unit ramp 0..1, the carrier a digital PWM peripheral actually uses."""

    def __init__(self, name, freq_hz):
        super().__init__(name)
        self.f = freq_hz

    def out(self, t, x, u):
        return (t * self.f) % 1.0


class Compare(Block):
    """Outputs hi when input 0 is above input 1, lo otherwise."""

    def __init__(self, name, hi=1.0, lo=0.0):
        super().__init__(name)
        self.hi, self.lo = hi, lo

    def out(self, t, x, u):
        return self.hi if u[0] >= u[1] else self.lo


class Integrator(Block):
    direct_feedthrough = False
    n_states = 1

    def __init__(self, name, x0=0.0):
        super().__init__(name)
        self.x0 = np.array([float(x0)])

    def out(self, t, x, u):
        return x[0]

    def deriv(self, t, x, u):
        return np.array([float(u[0])])


class StateSpace(Block):
    """dx/dt = A x + B u,  y = C x + D u. D = 0 keeps it free of feedthrough."""

    direct_feedthrough = False

    def __init__(self, name, A, B, C, D=None, x0=None):
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B = np.atleast_2d(np.asarray(B, dtype=float))
        self.C = np.atleast_2d(np.asarray(C, dtype=float))
        self.D = None if D is None else np.atleast_2d(np.asarray(D, float))
        # instance attribute only - mutating the class would break a model
        # that holds two state-space blocks of different order
        self.n_states = self.A.shape[0]
        super().__init__(name)
        self.direct_feedthrough = self.D is not None and np.any(self.D)
        self.x0 = np.zeros(self.n_states) if x0 is None else np.asarray(x0, float)

    def out(self, t, x, u):
        y = self.C @ x
        if self.D is not None:
            y = y + self.D @ np.asarray(u, dtype=float)
        return y[0] if y.size == 1 else y

    def deriv(self, t, x, u):
        return self.A @ x + (self.B @ np.asarray(u, dtype=float)).ravel()


class UnitDelay(Block):
    """One sample of delay, the 1/z block.

    Two state elements rather than one, and the reason matters. If the block
    simply captured its input at each sample instant, its output would change
    in the same instant and there would be no delay at all. Holding a pending
    value and promoting it on the next sample is what makes the output at step
    k equal to the input at step k-1.

    In a digital control loop this block is not decoration. It represents the
    time between the controller sampling and the new duty reaching the
    modulator, and it is usually the largest single source of phase lag in the
    loop.
    """

    direct_feedthrough = False

    def __init__(self, name, sample_time, x0=0.0):
        self.n_states = 2                      # [presented, pending]
        super().__init__(name)
        self.sample_time = float(sample_time)
        self.x0 = np.array([float(x0), float(x0)])

    def out(self, t, x, u):
        return x[0]

    def update(self, t, x, u):
        return np.array([x[1], float(u[0])])


class ZOH(Block):
    """Sample and hold: the block that turns a continuous signal into the
    number a microcontroller's ADC actually reads."""

    direct_feedthrough = False

    def __init__(self, name, sample_time, x0=0.0):
        self.n_states = 1
        super().__init__(name)
        self.sample_time = float(sample_time)
        self.x0 = np.array([float(x0)])

    def out(self, t, x, u):
        return x[0]

    def update(self, t, x, u):
        return np.array([float(u[0])])


class Scope(Block):
    """Collects named traces during the run."""

    def __init__(self, name, signals):
        super().__init__(name)
        self.signals = list(signals)     # block names to record
        self.t = []
        self.data = {s: [] for s in self.signals}

    def out(self, t, x, u):
        return 0.0

    def log(self, t, y):
        self.t.append(t)
        for s in self.signals:
            self.data[s].append(y[s])

    def trace(self, name):
        return np.asarray(self.data[name], dtype=float)

    def time(self):
        return np.asarray(self.t, dtype=float)
