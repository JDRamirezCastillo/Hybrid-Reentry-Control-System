"""
Hybrid LEO Re-entry Modeling and Control
=======================================

Python-centric MIL + Monte Carlo implementation aligned with:

- 3-DOF re-entry dynamics
- Exponential atmosphere, Kn/Ma/Re flow-regime classification
- POD ROM (100 modes)
- PID, L1 Adaptive, MPC, Q-learning controllers
- Hybrid supervisory switching with dwell-time
- Monte Carlo robustness campaigns

External dependencies:
    numpy
    cvxpy        (for MPC)
    torch        (optional, for ML-based classifier / Q-learning NN)

All physical parameters are configurable via Config.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Protocol, Tuple, List
import numpy as np

# Optional imports for MPC / ML; you can comment these out if not installed
try:
    import cvxpy as cp
except ImportError:
    cp = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


# ============================================================================
# 1. Global configuration
# ============================================================================

@dataclass
class Config:
    # Simulation / control timing
    dt: float = 0.01          # control step [s] (100 Hz)
    t_final: float = 300.0    # mission duration [s]

    # Vehicle parameters (approximate; adjust for your capsule)
    mass: float = 3000.0                 # kg
    reference_area: float = 15.0         # m^2
    nose_radius: float = 1.0             # m

    # Aerodynamic model coefficients (simple CL, CD vs alpha)
    cl_alpha: float = 20.0                # per rad
    cd0: float = 0.05
    cd_alpha2: float = 0.05               # quadratic in alpha

    # Control constraints
    alpha_min_deg: float = 0.0
    alpha_max_deg: float = 25.0
    bank_min_deg: float = -70.0
    bank_max_deg: float = 70.0

    # ROM configuration
    rom_modes: int = 100

    # MPC configuration (for a simple 2-state model)
    mpc_horizon: int = 15
    mpc_q_gamma: float = 1000.0
    mpc_q_gamma_dot: float = 1.0
    mpc_r_alpha: float = 0.1

    # L1 adaptive configuration
    l1_adaptation_gain: float = 5.0
    l1_filter_omega: float = 5.0  # rad/s cutoff

    # Q-learning configuration
    q_alpha_step_deg: float = 2.0
    q_num_actions: int = 5         # { -2, -1, 0, 1, 2 } * step
    q_gamma_err_bins: int = 11
    q_altitude_bins: int = 10
    q_learning_rate: float = 0.1
    q_discount: float = 0.99
    q_epsilon: float = 0.1

    # Hybrid switching
    min_dwell_time: float = 5.0   # s (approximate ADT proxy)
    e_pid_to_adaptive: float = 0.05  # rad
    e_adaptive_to_pid: float = 0.02  # rad
    mach_mpc_threshold: float = 15.0

    # Monte Carlo ranges
    rho_perturb_std: float = 0.30   # ±30% density variation


# ============================================================================
# 2. State, environment, atmosphere
# ============================================================================

@dataclass
class VehicleParams:
    mass: float
    reference_area: float
    nose_radius: float
    cl_alpha: float
    cd0: float
    cd_alpha2: float


@dataclass
class EnvironmentParams:
    re_earth: float = 6_371_000.0        # m
    mu_earth: float = 3.986e14           # m^3/s^2
    rho0: float = 1.225                  # kg/m^3
    H: float = 7200.0                    # m, scale height


@dataclass
class ControlInput:
    alpha: float   # rad
    bank: float    # rad


@dataclass
class State3DOF:
    r: float       # radial distance from center [m]
    v: float       # speed [m/s]
    gamma: float   # flight-path angle [rad]
    lon: float     # longitude [rad]
    lat: float     # latitude [rad]

    def as_vector(self) -> np.ndarray:
        return np.array([self.r, self.v, self.gamma, self.lon, self.lat], dtype=float)

    @classmethod
    def from_vector(cls, x: np.ndarray) -> "State3DOF":
        return cls(r=x[0], v=x[1], gamma=x[2], lon=x[3], lat=x[4])


class AtmosphereModel:
    def __init__(self, env: EnvironmentParams):
        self.env = env
        self.mean_free_path0 = 6.6e-8  # m, order-of-mag near sea level

    def density(self, h: float) -> float:
        return self.env.rho0 * np.exp(-h / self.env.H)

    def temperature(self, h: float) -> float:
        T0 = 288.15
        lapse = -0.0065
        return T0 + lapse * min(h, 11_000.0)

    def speed_of_sound(self, h: float) -> float:
        gamma = 1.4
        R = 287.0
        T = self.temperature(h)
        return np.sqrt(gamma * R * T)

    def mach(self, v: float, h: float) -> float:
        a = self.speed_of_sound(h)
        return v / max(a, 1e-6)

    def mean_free_path(self, rho: float) -> float:
        # crude inverse scaling with density
        return self.mean_free_path0 * (self.env.rho0 / max(rho, 1e-12))


# ============================================================================
# 3. 3-DOF re-entry dynamics
# ============================================================================

class ReentryDynamics3DOF:
    def __init__(self,
                 vehicle: VehicleParams,
                 env: EnvironmentParams,
                 atmosphere: AtmosphereModel):
        self.vehicle = vehicle
        self.env = env
        self.atmosphere = atmosphere

    def aerodynamic_coeffs(self, alpha: float, mach: float) -> Tuple[float, float]:
        cl = self.vehicle.cl_alpha * alpha
        cd = self.vehicle.cd0 + self.vehicle.cd_alpha2 * alpha**2
        return cl, cd

    def rhs(self, t: float, x: np.ndarray, u: ControlInput) -> np.ndarray:
        s = State3DOF.from_vector(x)
        h = s.r - self.env.re_earth
        rho = self.atmosphere.density(h)
        mach = self.atmosphere.mach(s.v, h)
        cl, cd = self.aerodynamic_coeffs(u.alpha, mach)

        q_dyn = 0.5 * rho * s.v**2
        L = q_dyn * cl * self.vehicle.reference_area
        D = q_dyn * cd * self.vehicle.reference_area

        g = self.env.mu_earth / s.r**2

        r_dot = s.v * np.sin(s.gamma)
        v_dot = -D / self.vehicle.mass - g * np.sin(s.gamma)
        gamma_dot = (L / (self.vehicle.mass * s.v)) + (s.v / s.r - g / s.v) * np.cos(s.gamma)

        lon_dot = (s.v * np.cos(s.gamma) * np.cos(u.bank)) / (s.r * max(np.cos(s.lat), 1e-6))
        lat_dot = (s.v * np.cos(s.gamma) * np.sin(u.bank)) / s.r

        return np.array([r_dot, v_dot, gamma_dot, lon_dot, lat_dot], dtype=float)

    def rk4_step(self, t: float, x: np.ndarray, u: ControlInput, dt: float) -> np.ndarray:
        k1 = self.rhs(t, x, u)
        k2 = self.rhs(t + 0.5 * dt, x + 0.5 * dt * k1, u)
        k3 = self.rhs(t + 0.5 * dt, x + 0.5 * dt * k2, u)
        k4 = self.rhs(t + dt, x + dt * k3, u)
        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


# ============================================================================
# 4. Flow-regime classifier
# ============================================================================

import enum

class FlowRegime(enum.Enum):
    CONTINUUM = "continuum"
    TRANSITIONAL = "transitional"
    FREE_MOLECULAR = "free_molecular"


@dataclass
class FlowRegimeClassifier:
    atmosphere: AtmosphereModel
    characteristic_length: float

    def knudsen(self, rho: float) -> float:
        lambda_mfp = self.atmosphere.mean_free_path(rho)
        return lambda_mfp / max(self.characteristic_length, 1e-6)

    def classify(self, rho: float) -> FlowRegime:
        kn = self.knudsen(rho)
        if kn < 0.01:
            return FlowRegime.CONTINUUM
        elif kn < 10.0:
            return FlowRegime.TRANSITIONAL
        else:
            return FlowRegime.FREE_MOLECULAR


# ============================================================================
# 5. POD-based ROM
# ============================================================================

@dataclass
class PODModel:
    mean_snapshot: np.ndarray
    modes: np.ndarray     # (n_state, r)
    singular_values: np.ndarray

    @classmethod
    def from_snapshots(cls, snapshots: np.ndarray, r: int) -> "PODModel":
        """
        snapshots: (n_state, n_snapshots)
        r: number of modes to keep (e.g., 100)
        """
        mean_snapshot = np.mean(snapshots, axis=1, keepdims=True)
        A = snapshots - mean_snapshot
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        modes = U[:, :r]
        singular_values = S[:r]
        return cls(mean_snapshot=mean_snapshot.flatten(),
                   modes=modes,
                   singular_values=singular_values)

    def project(self, snapshot: np.ndarray) -> np.ndarray:
        return self.modes.T @ (snapshot - self.mean_snapshot)

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        return self.mean_snapshot + self.modes @ coeffs


# ============================================================================
# 6. Controller base and implementations (PID, L1, MPC, Q-learning)
# ============================================================================

@dataclass
class ControllerConfig:
    dt: float
    alpha_limits: Tuple[float, float]
    bank_limits: Tuple[float, float]


class BaseController(Protocol):
    def reset(self) -> None: ...
    def step(self, t: float, state: State3DOF, ref: Dict[str, Any]) -> ControlInput: ...


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float


@dataclass
class PIDConfig(ControllerConfig):
    alpha_gains: PIDGains
    bank_gains: PIDGains


class PIDController:
    def __init__(self, cfg: PIDConfig):
        self.cfg = cfg
        self.int_errors: Dict[str, float] = {}
        self.prev_errors: Dict[str, float] = {}

    def reset(self) -> None:
        self.int_errors.clear()
        self.prev_errors.clear()

    def _pid_axis(self, name: str, e: float, gains: PIDGains) -> float:
        dt = self.cfg.dt
        i_prev = self.int_errors.get(name, 0.0)
        d_prev = self.prev_errors.get(name, 0.0)

        i_new = i_prev + e * dt
        d_new = (e - d_prev) / dt if dt > 0 else 0.0

        self.int_errors[name] = i_new
        self.prev_errors[name] = e

        return gains.kp * e + gains.ki * i_new + gains.kd * d_new

    def step(self, t: float, state: State3DOF, ref: Dict[str, Any]) -> ControlInput:
        gamma_ref = -1.5 * (3.14159 / 180.0)
        bank_ref = ref.get("bank_ref", 0.0)
        e_gamma = gamma_ref - state.gamma
        e_bank = bank_ref  # assume current bank = 0 in 3-DOF

        alpha_cmd = self._pid_axis("alpha", e_gamma, self.cfg.alpha_gains)
        bank_cmd = self._pid_axis("bank", e_bank, self.cfg.bank_gains)

        alpha_cmd = np.clip(alpha_cmd, *self.cfg.alpha_limits)
        bank_cmd = np.clip(bank_cmd, *self.cfg.bank_limits)
        return ControlInput(alpha=float(alpha_cmd), bank=float(bank_cmd))


# ---------- L1 adaptive (simple SISO variant) ----------

@dataclass
class L1Config(ControllerConfig):
    adaptation_gain: float
    filter_omega: float


class L1AdaptiveController:
    """
    Simple L1-like adaptive controller on gamma channel.
    This is a simplified variant: reference model is first-order,
    adaptation on an uncertain gain, and low-pass filtered control.
    """

    def __init__(self, cfg: L1Config):
        self.cfg = cfg
        self.theta_hat = 0.0
        self.u_filtered = 0.0

    def reset(self) -> None:
        self.theta_hat = 0.0
        self.u_filtered = 0.0

    def step(self, t: float, state: State3DOF, ref: Dict[str, Any]) -> ControlInput:
        gamma_ref = ref.get("gamma_ref", 0.0)
        e = gamma_ref - state.gamma

        # Adaptation law (very simple gradient-type)
        self.theta_hat += self.cfg.adaptation_gain * e * self.cfg.dt

        # Raw control (proportional + adaptive term)
        k_p = 2.0
        u_raw = k_p * e + self.theta_hat

        # Low-pass filter (first-order)
        alpha = np.exp(-self.cfg.filter_omega * self.cfg.dt)
        self.u_filtered = alpha * self.u_filtered + (1 - alpha) * u_raw

        alpha_cmd = np.clip(self.u_filtered, *self.cfg.alpha_limits)
        bank_cmd = 0.0  # keep bank neutral here
        return ControlInput(alpha=float(alpha_cmd), bank=float(bank_cmd))


# ---------- MPC controller (linearized 2-state example) ----------

@dataclass
class MPCConfig(ControllerConfig):
    horizon: int
    q_gamma: float
    q_gamma_dot: float
    r_alpha: float


class MPCController:
    """
    Simple linear MPC on [gamma, gamma_dot] with scalar input alpha.

    Uses cvxpy; for real-time you would move to a pre-compiled QP solver.
    """

    def __init__(self, cfg: MPCConfig):
        if cp is None:
            raise RuntimeError("cvxpy is required for MPCController")
        self.cfg = cfg
        # Linearized model (example; you can replace with your own)
        self.A = np.array([[1.0, self.cfg.dt],
                           [0.0, 1.0]], dtype=float)
        self.B = np.array([[0.0],
                           [self.cfg.dt]], dtype=float)

        # build cvxpy problem structure
        N = cfg.horizon
        nx, nu = 2, 1
        self.x_var = cp.Variable((nx, N+1))
        self.u_var = cp.Variable((nu, N))

        self.Q = np.diag([cfg.q_gamma, cfg.q_gamma_dot])
        self.R = np.array([[cfg.r_alpha]])

        self.constraints = []
        self.cost = 0
        self.x0_param = cp.Parameter(nx)
        self.gamma_ref_param = cp.Parameter(N)

        for k in range(N):
            self.cost += cp.quad_form(self.x_var[:, k] - cp.reshape(cp.vstack([self.gamma_ref_param[k], 0.0]), (2,), order='C'), self.Q)
            self.cost += cp.quad_form(self.u_var[:, k], self.R)
            self.constraints += [self.x_var[:, k+1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k]]

            self.constraints += [
                self.u_var[0, k] >= self.cfg.alpha_limits[0],
                self.u_var[0, k] <= self.cfg.alpha_limits[1],
            ]

        self.prob = cp.Problem(cp.Minimize(self.cost), self.constraints)

    def reset(self) -> None:
        pass

    def step(self, t: float, state: State3DOF, ref: Dict[str, Any]) -> ControlInput:
        gamma_ref = ref.get("gamma_ref", 0.0)

        # Approximate gamma_dot by finite difference of ref (could be improved)
        gamma_dot_est = 0.0
        x0 = np.array([state.gamma, gamma_dot_est], dtype=float)
        self.x0_param.value = x0

        N = self.cfg.horizon
        self.gamma_ref_param.value = np.full(N, gamma_ref, dtype=float)

        # Initial condition constraint
        self.constraints.insert(0, self.x_var[:, 0] == self.x0_param)

        self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        u0 = float(self.u_var.value[0, 0])
        alpha_cmd = np.clip(u0, *self.cfg.alpha_limits)
        bank_cmd = 0.0
        # remove the initial condition constraint to avoid growth
        self.constraints.pop(0)
        return ControlInput(alpha=alpha_cmd, bank=bank_cmd)


# ---------- Q-learning controller (discrete actions) ----------

@dataclass
class QConfig(ControllerConfig):
    alpha_step_deg: float
    num_actions: int
    gamma_err_bins: int
    altitude_bins: int
    lr: float
    gamma_discount: float
    epsilon: float


class QLearningController:
    """
    Tabular Q-learning over discretized (gamma error, altitude) with discrete alpha actions.
    Intended for offline training in MIL, then frozen in flight.
    """

    def __init__(self, cfg: QConfig, env_params: EnvironmentParams):
        self.cfg = cfg
        self.env = env_params
        self.actions = np.linspace(
            - (cfg.num_actions // 2) * np.deg2rad(cfg.alpha_step_deg),
            + (cfg.num_actions // 2) * np.deg2rad(cfg.alpha_step_deg),
            cfg.num_actions
        )
        self.Q = np.zeros((cfg.gamma_err_bins, cfg.altitude_bins, cfg.num_actions), dtype=float)

    def reset(self) -> None:
        pass

    def _discretize_state(self, gamma_err: float, altitude: float) -> Tuple[int, int]:
        # gamma_err in [-20°, 20°]
        ge = np.clip(gamma_err, np.deg2rad(-20), np.deg2rad(20))
        ge_norm = (ge - np.deg2rad(-20)) / np.deg2rad(40)
        ge_bin = int(ge_norm * (self.cfg.gamma_err_bins - 1))

        # altitude in [0, 120 km]
        h = np.clip(altitude, 0.0, 120_000.0)
        h_norm = h / 120_000.0
        h_bin = int(h_norm * (self.cfg.altitude_bins - 1))

        return ge_bin, h_bin

    def _select_action_index(self, s: Tuple[int, int]) -> int:
        if np.random.rand() < self.cfg.epsilon:
            return np.random.randint(0, self.cfg.num_actions)
        return int(np.argmax(self.Q[s[0], s[1], :]))

    def step(self, t: float, state: State3DOF, ref: Dict[str, Any]) -> ControlInput:
        gamma_ref = ref.get("gamma_ref", 0.0)
        e_gamma = gamma_ref - state.gamma
        altitude = state.r - self.env.re_earth
        s = self._discretize_state(e_gamma, altitude)
        a_idx = self._select_action_index(s)
        alpha_cmd = np.clip(self.actions[a_idx], *self.cfg.alpha_limits)
        bank_cmd = 0.0
        return ControlInput(alpha=float(alpha_cmd), bank=float(bank_cmd))

    # Training function can be added here to interact with a MIL environment.


# ============================================================================
# 7. Hybrid supervisory switching
# ============================================================================

@dataclass
class HybridConfig:
    dt: float
    min_dwell_time: float
    e_pid_to_adaptive: float
    e_adaptive_to_pid: float
    mach_mpc_threshold: float


class HybridSupervisor:
    """
    Supervisory controller switching between PID, L1, MPC, and Q-learning
    based on error, Mach, and flow regime, with simple dwell-time enforcement.
    """

    def __init__(self,
                 cfg: HybridConfig,
                 pid: BaseController,
                 l1: BaseController,
                 mpc: BaseController,
                 qlearn: BaseController,
                 classifier: FlowRegimeClassifier,
                 env: EnvironmentParams,
                 atmosphere: AtmosphereModel):
        self.cfg = cfg
        self.pid = pid
        self.l1 = l1
        self.mpc = mpc
        self.qlearn = qlearn
        self.classifier = classifier
        self.env = env
        self.atmosphere = atmosphere

        self.active_ctrl: BaseController = pid
        self.last_switch_time: float = 0.0

    def reset(self) -> None:
        self.pid.reset()
        self.l1.reset()
        self.mpc.reset()
        self.qlearn.reset()
        self.active_ctrl = self.pid
        self.last_switch_time = 0.0

    def _can_switch(self, t: float) -> bool:
        return (t - self.last_switch_time) >= self.cfg.min_dwell_time

    def _select_mode(self, t: float, state: State3DOF, ref: Dict[str, Any]) -> None:
        h = state.r - self.env.re_earth
        rho = self.atmosphere.density(h)
        mach = self.atmosphere.mach(state.v, h)
        regime = self.classifier.classify(rho)

        gamma_ref = ref.get("gamma_ref", 0.0)
        e_gamma = abs(gamma_ref - state.gamma)

        if not self._can_switch(t):
            return

        # Simple logic (extend as needed):
        # - PID in continuum, low error, subcritical Mach
        # - L1 in transitional regime or larger error
        # - MPC when Mach is high
        # - Q-learning reserved for terminal phase (altitude < 40 km)
        altitude = h

        if altitude < 40_000.0:
            # terminal phase: let Q-learning refine landing
            self.active_ctrl = self.qlearn
            self.last_switch_time = t
            return

        if mach > self.cfg.mach_mpc_threshold and regime == FlowRegime.CONTINUUM:
            self.active_ctrl = self.mpc
            self.last_switch_time = t
            return

        if regime != FlowRegime.CONTINUUM or e_gamma > self.cfg.e_pid_to_adaptive:
            self.active_ctrl = self.l1
            self.last_switch_time = t
            return

        if e_gamma < self.cfg.e_adaptive_to_pid and regime == FlowRegime.CONTINUUM:
            self.active_ctrl = self.pid
            self.last_switch_time = t

    def step(self, t: float, state: State3DOF, ref: Dict[str, Any]) -> ControlInput:
        self._select_mode(t, state, ref)
        return self.active_ctrl.step(t, state, ref)


# ============================================================================
# 8. MIL simulator and metrics
# ============================================================================

@dataclass
class MILScenario:
    t0: float
    tf: float
    dt: float
    initial_state: State3DOF
    vehicle: VehicleParams
    ref_profile: Callable[[float], Dict[str, Any]]


@dataclass
class MILResult:
    times: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    metrics: Dict[str, float]


def rms_error(y: np.ndarray, y_ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - y_ref) ** 2)))


class MILSimulator:
    def __init__(self,
                 controller: BaseController,
                 env: EnvironmentParams,
                 atmosphere: AtmosphereModel):
        self.controller = controller
        self.env = env
        self.atmosphere = atmosphere

    def run(self, scenario: MILScenario) -> MILResult:
        dyn = ReentryDynamics3DOF(
            vehicle=scenario.vehicle,
            env=self.env,
            atmosphere=self.atmosphere
        )
        t = scenario.t0
        x = scenario.initial_state.as_vector()
        times: List[float] = []
        states: List[np.ndarray] = []
        controls: List[np.ndarray] = []

        self.controller.reset()

        while t <= scenario.tf:
            s = State3DOF.from_vector(x)
            ref = scenario.ref_profile(t)
            u = self.controller.step(t, s, ref)
            x = dyn.rk4_step(t, x, u, scenario.dt)

            times.append(t)
            states.append(x.copy())
            controls.append(np.array([u.alpha, u.bank], dtype=float))
            t += scenario.dt

        times_arr = np.array(times)
        states_arr = np.vstack(states)
        controls_arr = np.vstack(controls)

        gamma = states_arr[:, 2]
        gamma_ref = np.array([scenario.ref_profile(ti)["gamma_ref"] for ti in times_arr])
        rms_gamma = rms_error(gamma, gamma_ref)

        metrics = {"rms_gamma": rms_gamma}
        return MILResult(times=times_arr, states=states_arr, controls=controls_arr, metrics=metrics)


# ============================================================================
# 9. Monte Carlo campaign
# ============================================================================

@dataclass
class UncertaintySpec:
    name: str
    sampler: Callable[[np.random.Generator], float]


@dataclass
class MonteCarloResult:
    metrics: Dict[str, np.ndarray]


class MonteCarloCampaign:
    def __init__(self,
                 simulator_factory: Callable[[Dict[str, Any]], MILSimulator],
                 base_scenario: MILScenario,
                 uncertainties: List[UncertaintySpec],
                 n_runs: int):
        self.simulator_factory = simulator_factory
        self.base_scenario = base_scenario
        self.uncertainties = uncertainties
        self.n_runs = n_runs

    def run(self, seed: int = 0) -> MonteCarloResult:
        rng = np.random.default_rng(seed)
        rms_list = []

        for i in range(self.n_runs):
            params: Dict[str, Any] = {}
            for spec in self.uncertainties:
                params[spec.name] = spec.sampler(rng)
            sim = self.simulator_factory(params)

            # For now, we don't modify the scenario itself; you can inject
            # params into vehicle, atmosphere, or initial state here.
            result = sim.run(self.base_scenario)
            rms_list.append(result.metrics["rms_gamma"])

        return MonteCarloResult(metrics={"rms_gamma": np.array(rms_list)})


# ============================================================================
# 10. Helper to build a default hybrid system and scenario
# ============================================================================

def build_default_system(cfg: Config) -> Tuple[HybridSupervisor, MILScenario,
                                               EnvironmentParams, AtmosphereModel]:
    env = EnvironmentParams()
    atmosphere = AtmosphereModel(env)
    vehicle = VehicleParams(
        mass=cfg.mass,
        reference_area=cfg.reference_area,
        nose_radius=cfg.nose_radius,
        cl_alpha=cfg.cl_alpha,
        cd0=cfg.cd0,
        cd_alpha2=cfg.cd_alpha2
    )

    alpha_limits = (np.deg2rad(cfg.alpha_min_deg), np.deg2rad(cfg.alpha_max_deg))
    bank_limits = (np.deg2rad(cfg.bank_min_deg), np.deg2rad(cfg.bank_max_deg))

    pid_cfg = PIDConfig(
        dt=cfg.dt,
        alpha_limits=alpha_limits,
        bank_limits=bank_limits,
        alpha_gains=PIDGains(50.0, 5.0, 10.0),
        bank_gains=PIDGains(0.2, 0.0, 0.05)
    )
    pid = PIDController(pid_cfg)

    l1_cfg = L1Config(
        dt=cfg.dt,
        alpha_limits=alpha_limits,
        bank_limits=bank_limits,
        adaptation_gain=cfg.l1_adaptation_gain,
        filter_omega=cfg.l1_filter_omega
    )
    l1 = L1AdaptiveController(l1_cfg)

    mpc_cfg = MPCConfig(
        dt=cfg.dt,
        alpha_limits=alpha_limits,
        bank_limits=bank_limits,
        horizon=cfg.mpc_horizon,
        q_gamma=500.0,
        q_gamma_dot=cfg.mpc_q_gamma_dot,
        r_alpha=cfg.mpc_r_alpha
    )
    mpc = MPCController(mpc_cfg)

    q_cfg = QConfig(
        dt=cfg.dt,
        alpha_limits=alpha_limits,
        bank_limits=bank_limits,
        alpha_step_deg=cfg.q_alpha_step_deg,
        num_actions=cfg.q_num_actions,
        gamma_err_bins=cfg.q_gamma_err_bins,
        altitude_bins=cfg.q_altitude_bins,
        lr=cfg.q_learning_rate,
        gamma_discount=cfg.q_discount,
        epsilon=cfg.q_epsilon
    )
    qlearn = QLearningController(q_cfg, env)

    classifier = FlowRegimeClassifier(
        atmosphere=atmosphere,
        characteristic_length=cfg.nose_radius
    )

    hybrid_cfg = HybridConfig(
        dt=cfg.dt,
        min_dwell_time=cfg.min_dwell_time,
        e_pid_to_adaptive=cfg.e_pid_to_adaptive,
        e_adaptive_to_pid=cfg.e_adaptive_to_pid,
        mach_mpc_threshold=cfg.mach_mpc_threshold
    )

    supervisor = HybridSupervisor(
        cfg=hybrid_cfg,
        pid=pid,
        l1=l1,
        mpc=mpc,
        qlearn=qlearn,
        classifier=classifier,
        env=env,
        atmosphere=atmosphere
    )

    # Nominal entry: 120 km, 7.8 km/s, -6 deg γ, arbitrary lon/lat
    r0 = env.re_earth + 120_000.0
    v0 = 7_800.0
    gamma0 = np.deg2rad(-6.0)
    lon0, lat0 = 0.0, 0.0
    initial_state = State3DOF(r=r0, v=v0, gamma=gamma0, lon=lon0, lat=lat0)

    def ref_profile(t: float) -> Dict[str, Any]:
        # simple constant gamma reference; you can replace with your
        # optimized profile (e.g. for thermal management)
        return {"gamma_ref": gamma0, "bank_ref": 0.0}

    scenario = MILScenario(
        t0=0.0,
        tf=cfg.t_final,
        dt=cfg.dt,
        initial_state=initial_state,
        vehicle=vehicle,
        ref_profile=ref_profile
    )
    return supervisor, scenario, env, atmosphere


if __name__ == "__main__":
    # Example: run a single MIL simulation with the hybrid controller
    cfg = Config()
    supervisor, scenario, env, atmosphere = build_default_system(cfg)
    mil = MILSimulator(supervisor, env, atmosphere)
    result = mil.run(scenario)
    print("RMS gamma error [rad]:", result.metrics["rms_gamma"])