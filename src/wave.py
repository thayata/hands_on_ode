import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class WaveAnimation:
    def __init__(self, omega=2 * np.pi, wavelength=5, x_max=10, samples=400):
        self.omega = omega
        self.k = 2 * np.pi / wavelength
        self.x = np.linspace(0, x_max, samples)
        self.wave_func = self._default_wave

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(self.x.min(), self.x.max())
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("sin(ωt - kx)")
        self.ax.set_title("Propagation of sin(ωt - kx)")

        (self.line,) = self.ax.plot([], [], lw=2, color="tab:blue")

    def _default_wave(self, t, x):
        return np.sin(self.omega * t - self.k * x)

    def set_wave_function(self, func):
        if not callable(func):
            raise TypeError("wave function must be callable")
        test = np.asarray(func(0.0, self.x))
        if test.shape != self.x.shape:
            raise ValueError("wave function must return an array matching x")
        self.wave_func = func

    def init_wave(self):
        self.line.set_data(self.x, np.zeros_like(self.x))
        return (self.line,)

    def update_wave(self, t):
        y = np.asarray(self.wave_func(t, self.x))
        self.line.set_data(self.x, y)
        return (self.line,)

    def to_html(self, interval=50, repeat=True):
        period = 2 * np.pi / self.omega
        frames = np.linspace(0, 2 * period, 200)
        animation = FuncAnimation(
            self.fig,
            self.update_wave,
            frames=frames,
            init_func=self.init_wave,
            interval=interval,
            blit=True,
            repeat=repeat,
        )
        plt.close(self.fig)
        return HTML(animation.to_jshtml())
    
class Advection:
    def __init__(
            self,
            velocity=1.0,
            x_min=0.0,
            x_max=10.0,
            samples=400,
            dt=None,
            scheme="central",
        ):
            if samples < 2:
                raise ValueError("samples must be at least 2.")
            self.c = float(velocity)
            self.x = np.linspace(x_min, x_max, samples, endpoint=False)
            self.dx = (x_max - x_min) / samples
            if dt is None:
                c_abs = abs(self.c)
                self.dt = 0.4 * self.dx / c_abs if c_abs > 0 else 0.1 * self.dx
            else:
                self.dt = float(dt)
            #if abs(self.c) * self.dt / self.dx >= 1.0:
            #    raise ValueError("CFL condition violated: |c| * dt / dx must be < 1.")
            self.u = np.zeros_like(self.x, dtype=float)
            self.time = 0.0

            fig, ax = plt.subplots()
            self.fig = fig
            self.ax = ax
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-1.0, 1.0)
            ax.set_xlabel("x")
            ax.set_ylabel("u(x, t)")
            ax.set_title("Solution of the advection equation")
            (self.line,) = ax.plot(self.x, self.u, color="tab:orange", lw=2)

            self.set_spatial_scheme(scheme)

    def set_spatial_scheme(self, scheme):
            allowed = {"forward", "backward", "central", "upwind"}
            if scheme not in allowed:
                raise ValueError(f"scheme must be one of {sorted(allowed)}")
            self.scheme = scheme

    def set_initial_condition(self, func):
            if not callable(func):
                raise TypeError("initial condition must be callable")
            values = np.asarray(func(self.x), dtype=float)
            if values.shape != self.x.shape:
                raise ValueError("initial condition must match spatial grid")
            self.u = values.copy()
            self.time = 0.0
            span = values.max() - values.min()
            pad = 0.1 * span if span > 0 else 0.1
            self.ax.set_ylim(values.min() - pad, values.max() + pad)
            self.line.set_data(self.x, self.u)

    def init_profile(self):
            self.line.set_data(self.x, self.u)
            return (self.line,)

    def _compute_gradient(self):
            if self.scheme == "forward":
                return (np.roll(self.u, -1) - self.u) / self.dx
            if self.scheme == "backward":
                return (self.u - np.roll(self.u, 1)) / self.dx
            if self.scheme == "central":
                return (np.roll(self.u, -1) - np.roll(self.u, 1)) / (2.0 * self.dx)
            if self.c >= 0.0:
                return (self.u - np.roll(self.u, 1)) / self.dx
            return (np.roll(self.u, -1) - self.u) / self.dx

    def _step(self):
            if self.c == 0.0:
                self.time += self.dt
                return
            grad = self._compute_gradient()
            self.u = self.u - self.c * self.dt * grad
            self.time += self.dt

    def update_profile(self, _):
            self._step()
            self.line.set_data(self.x, self.u)
            return (self.line,)

    def to_html(self, steps=200, interval=50, repeat=True):
            animation = FuncAnimation(
                self.fig,
                self.update_profile,
                frames=range(steps),
                init_func=self.init_profile,
                interval=interval,
                blit=True,
                repeat=repeat,
            )
            plt.close(self.fig)
            return HTML(animation.to_jshtml())
        
    class BackwardAdvection:
        def __init__(self, velocity=1.0, x_min=0.0, x_max=10.0, samples=400, dt=None):
                if samples < 3:
                    raise ValueError("central difference requires at least three spatial points")
                self.c = float(velocity)
                self.bounds = (float(x_min), float(x_max))
                self.nodes = np.linspace(self.bounds[0], self.bounds[1], int(samples), endpoint=False)
                self.dx = (self.bounds[1] - self.bounds[0]) / self.nodes.size
                self.dt = self._select_dt(dt)
                self.state = np.zeros_like(self.nodes)
                self.time = 0.0
                self._assemble_operator()

                fig, ax = plt.subplots()
                self.figure = fig
                self.axes = ax
                ax.set(xlim=self.bounds, ylim=(-1.0, 1.0), xlabel="x", ylabel="u(x, t)", title="Backward Euler (time) with Central Difference (space)")
                (self.line,) = ax.plot(self.nodes, self.state, color="tab:green", lw=2)

        def _select_dt(self, dt):
                if dt is not None:
                    return float(dt)
                if self.c == 0.0:
                    return 0.1 * self.dx
                return 0.4 * self.dx / abs(self.c)

        def _assemble_operator(self):
                n = self.nodes.size
                coeff = self.c * self.dt / (2.0 * self.dx)
                mat = np.eye(n)
                idx = np.arange(n)
                mat[idx, (idx + 1) % n] += coeff
                mat[idx, (idx - 1) % n] -= coeff
                self._system = mat
                self._advance = np.linalg.inv(mat)

        def set_initial_condition(self, func):
                if not callable(func):
                    raise TypeError("initial condition must be callable")
                data = np.asarray(func(self.nodes), dtype=float)
                if data.shape != self.nodes.shape:
                    raise ValueError("initial condition must match spatial grid")
                self.state = data.copy()
                self.time = 0.0
                lower = float(data.min())
                upper = float(data.max())
                spread = upper - lower
                padding = 0.1 * spread if spread else 0.1
                self.axes.set_ylim(lower - padding, upper + padding)
                self.line.set_data(self.nodes, self.state)

        def init_profile(self):
                self.line.set_data(self.nodes, self.state)
                return (self.line,)

        def _step(self):
                self.state = self._advance @ self.state
                self.time += self.dt

        def update_profile(self, _frame=None):
                self._step()
                self.line.set_data(self.nodes, self.state)
                return (self.line,)

        def to_html(self, steps=200, interval=50, repeat=True):
                animation = FuncAnimation(
                    self.figure,
                    self.update_profile,
                    frames=np.arange(steps),
                    init_func=self.init_profile,
                    interval=interval,
                    blit=True,
                    repeat=repeat,
                )
                plt.close(self.figure)
                return HTML(animation.to_jshtml())