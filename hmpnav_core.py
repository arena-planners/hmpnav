"""ROS-free port of the HMPNav MPPI controller (fluentrobotics/HMPNav, BSD-3).

Upstream sources this mirrors (commit 8b23db2):
  controller_config.py  constants (DT, horizon, samples, Q_*, SIGMA_*)
  node.py               MPPI(...) construction, terminal_cost(), dynamics(), turn-to-goal
  cv.py                 construct_cv_predictions, goal_cost_terminal, obstacle_cost_terminal,
                        collision_avoidance_cost_terminal
  pytorch_mppi.MPPI     sampling / importance weighting (MIT, UM-ARM-Lab); re-implemented
                        here in ~40 lines so the planner venv needs only torch + numpy.

Deliberate deviations from upstream, all for the arena setting:
  * control box is the Jackal's (v in [0, V_MAX], omega in [-W_MAX, W_MAX]) instead of the
    Stretch's ([0,1] x [-1,1], then v clipped to 0.3 on publish).
  * static obstacles come from the robot's laser scan, rasterised into a robot-centred
    inflated occupancy grid, instead of hard-coded lab polygons (STATIC_OBS / workstations).
    A rollout state inside the grid costs like a workstation hit upstream (1000 * 1000).
  * the pedestrian Gaussian is summed over agents and scaled by 1/ACTIVE_AGENTS (=2)
    instead of averaged over agents; identical for the paper's 2-person study, but does
    not dilute the nearest person's cost in crowds of 10-20.
  * the nominal sequence starts at zeros (pytorch_mppi draws it from the noise dist).
  * q_obs_scale / sigma_scale (default 1 = upstream) scale Q_OBS and SIGMA_{H,S,R}; the arena
    adapter defaults to 10 / 2 (see planner.py _PROFILES).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

# --- controller_config.py -------------------------------------------------------------
DT = 0.4
HISTORY_LENGTH = 7
WINDOW_LENGTH = 20
PREDICTION_LENGTH = WINDOW_LENGTH - HISTORY_LENGTH - 1  # 12 CV points
HORIZON = PREDICTION_LENGTH - 1  # 11 MPPI steps (4.4 s)
PREDICTION_HORIZON = DT * PREDICTION_LENGTH  # 4.8 s, used for the goal normaliser
ACTIVE_AGENTS = 2
SIGMA_H = 0.3
SIGMA_S = 0.3
SIGMA_R = 0.6
Q_OBS = 10e3
Q_GOAL = 1.0
NUM_SAMPLES = 2000
NOISE_VAR = 2.0  # node.py: cov = diag(2, 2)
LAMBDA = 1.0  # pytorch_mppi default temperature (upstream does not override it)
TURN_DONE = math.pi / 64  # node.py turning_to_center exit threshold
TURN_SLOW_BAND = math.pi / 6
TURN_W_FAST = 1.5
TURN_W_SLOW = 0.2

# cv.py set_params(): normalised weights
Q_GOAL_N = Q_GOAL / (2.0 / PREDICTION_HORIZON) ** 2
Q_OBS_N = Q_OBS / 0.5**2
COLLISION_STATE_COST = 1000.0 * 1000.0  # terminal_cost: 1000 * cc, cc = 1000 per workstation hit


@dataclass
class Limits:
    v_max: float = 0.5
    w_max: float = 1.5


def normalize_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def dynamics(s: torch.Tensor, a: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """utils.dynamics / node.dynamics: exact unicycle arc over dt. s (K,3), a (K,2)."""
    v, w = a[:, 0], a[:, 1]
    d_theta = w * dt
    straight = w.abs() < 1e-6
    w_safe = torch.where(straight, torch.ones_like(w), w)
    r = v / w_safe
    ex = torch.where(straight, v * dt, r * torch.sin(d_theta))
    ey = torch.where(straight, torch.zeros_like(v), r * (1.0 - torch.cos(d_theta)))
    et = torch.where(straight, torch.zeros_like(v), d_theta)
    c, sn = torch.cos(s[:, 2]), torch.sin(s[:, 2])
    out = torch.empty_like(s)
    out[:, 0] = s[:, 0] + ex * c - ey * sn
    out[:, 1] = s[:, 1] + ex * sn + ey * c
    out[:, 2] = normalize_angle(s[:, 2] + et)
    return out


class LocalGrid:
    """Robot-centred boolean occupancy grid from world-frame obstacle points, pre-inflated."""

    def __init__(self, half_size: float = 3.2, resolution: float = 0.05) -> None:
        self.res = resolution
        self.n = int(round(2 * half_size / resolution))
        self.half = self.n * resolution / 2.0
        self._offsets: dict[float, np.ndarray] = {}

    def _disc(self, radius: float) -> np.ndarray:
        if radius not in self._offsets:
            k = int(math.ceil(radius / self.res))
            yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
            m = (xx * xx + yy * yy) * self.res**2 <= radius * radius
            self._offsets[radius] = np.stack([yy[m], xx[m]], axis=1)
        return self._offsets[radius]

    def build(self, cx: float, cy: float, points: np.ndarray, inflation: float) -> np.ndarray:
        grid = np.zeros((self.n, self.n), dtype=bool)
        self.ox, self.oy = cx - self.half, cy - self.half
        if points.size == 0:
            return grid
        rows = np.floor((points[:, 1] - self.oy) / self.res).astype(np.int64)
        cols = np.floor((points[:, 0] - self.ox) / self.res).astype(np.int64)
        off = self._disc(inflation)
        rr = (rows[:, None] + off[None, :, 0]).ravel()
        cc = (cols[:, None] + off[None, :, 1]).ravel()
        ok = (rr >= 0) & (rr < self.n) & (cc >= 0) & (cc < self.n)
        grid[rr[ok], cc[ok]] = True
        return grid

    def lookup(self, grid: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """Occupancy at world xy (..., 2); outside the window reads free."""
        col = torch.floor((xy[..., 0] - self.ox) / self.res).long()
        row = torch.floor((xy[..., 1] - self.oy) / self.res).long()
        inside = (row >= 0) & (row < self.n) & (col >= 0) & (col < self.n)
        flat = (row.clamp(0, self.n - 1) * self.n + col.clamp(0, self.n - 1)).reshape(-1)
        occ = grid.reshape(-1)[flat].reshape(inside.shape)
        return occ & inside


class CVPredictor:
    """cv.py construct_cv_predictions: displacement over the last DT, extrapolated PREDICTION_LENGTH steps.

    Keeps a per-id position history (one entry per bridge tick). With a full DT window the
    displacement is the finite difference over DT exactly as upstream; with a shorter history it
    is rescaled; with a single sample it falls back to the pedestrian twist (arena_peds vx, vy).
    """

    def __init__(self, tick_dt: float) -> None:
        self.set_tick(tick_dt)
        self.hist: dict[int, deque] = {}
        self.tick = 0

    def set_tick(self, tick_dt: float) -> None:
        self.tick_dt = float(tick_dt)
        self.n_dt = max(1, int(round(DT / self.tick_dt)))

    def reset(self) -> None:
        self.hist.clear()
        self.tick = 0

    def update(self, peds: np.ndarray) -> np.ndarray:
        """peds (H,>=3) [id,x,y,(vx,vy)] -> predictions (PREDICTION_LENGTH+1, H, 2); index 0 = now."""
        self.tick += 1
        H = peds.shape[0]
        seen = set()
        disp = np.zeros((H, 2), dtype=np.float64)
        for i in range(H):
            pid = int(peds[i, 0])
            seen.add(pid)
            d = self.hist.get(pid)
            if d is None:
                d = deque(maxlen=self.n_dt + 1)
                self.hist[pid] = d
            d.append((self.tick, float(peds[i, 1]), float(peds[i, 2])))
            t0, x0, y0 = d[0]
            span = self.tick - t0
            if span > 0:
                disp[i] = ((peds[i, 1] - x0) * self.n_dt / span, (peds[i, 2] - y0) * self.n_dt / span)
            elif peds.shape[1] >= 5:
                disp[i] = (peds[i, 3] * DT, peds[i, 4] * DT)
        for pid in [p for p in self.hist if p not in seen]:
            del self.hist[pid]
        steps = np.arange(0, PREDICTION_LENGTH + 1, dtype=np.float64)[:, None, None]
        return peds[None, :, 1:3].astype(np.float64) + steps * disp[None]


class HMPNavMPPI:
    def __init__(
        self,
        num_samples: int = NUM_SAMPLES,
        limits: Limits | None = None,
        device: str = "cpu",
        robot_radius: float = 0.267,
        clearance: float = 0.10,
        seed: int = 0,
        q_obs_scale: float = 1.0,
        sigma_scale: float = 1.0,
    ) -> None:
        self.K = int(num_samples)
        self.T = HORIZON
        self.lim = limits or Limits()
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.u_min = torch.tensor([0.0, -self.lim.w_max], device=self.device, dtype=self.dtype)
        self.u_max = torch.tensor([self.lim.v_max, self.lim.w_max], device=self.device, dtype=self.dtype)
        self.noise_std = math.sqrt(NOISE_VAR)
        self.sigma_inv = torch.eye(2, device=self.device, dtype=self.dtype) / NOISE_VAR
        self.inflation = robot_radius + clearance
        self.q_obs = Q_OBS_N * float(q_obs_scale)
        self.sig_h, self.sig_s, self.sig_r = (SIGMA_H * sigma_scale, SIGMA_S * sigma_scale, SIGMA_R * sigma_scale)
        self.grid = LocalGrid(half_size=self.lim.v_max * DT * self.T + self.inflation + 0.6)
        self.gen = torch.Generator(device=self.device)
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self.gen.manual_seed(int(seed))
        self.U = torch.zeros(self.T, 2, device=self.device, dtype=self.dtype)
        self.u_init = torch.zeros(2, device=self.device, dtype=self.dtype)
        self.turning = True
        self.turn_ticks = 0
        self.last_states: torch.Tensor | None = None
        self.last_cost: torch.Tensor | None = None

    # ---- costs (cv.py, terminal-cost variant: USE_TERMINAL_COST = True) -------------------
    def _goal_cost(self, S: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(goal[None, None, :] - S[:, :, :2], dim=2)
        return Q_GOAL_N * dist.sum(dim=1) ** 2

    def _ped_cost(self, S: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """obstacle_cost_terminal with S=1 mode; P (T+1, H, 2) aligned so P[k] ~ S[:, k]."""
        T = S.shape[1]
        dx = S[:, :, None, 0] - P[None, :T, :, 0]
        dy = S[:, :, None, 1] - P[None, :T, :, 1]
        vx = (P[1 : T + 1, :, 0] - P[:T, :, 0]) / DT
        vy = (P[1 : T + 1, :, 1] - P[:T, :, 1]) / DT
        th = torch.atan2(vy, vx)[None]
        static = (torch.sqrt(vx * vx + vy * vy) < 0.01)[None]
        alpha = torch.atan2(dy, dx) - th + math.pi / 2.0
        alpha = (torch.remainder(alpha + math.pi, 2 * math.pi) - math.pi) <= 0
        sig = torch.where(alpha, self.sig_r, self.sig_h)
        sig = torch.where(static, 1.0, sig)
        sig_s = torch.where(static, 1.0, self.sig_s).expand_as(sig)
        c2, s2, sn2 = torch.cos(th) ** 2, torch.sin(th) ** 2, torch.sin(2 * th)
        a = c2 / (2 * sig**2) + s2 / (2 * sig_s**2)
        b = sn2 / (4 * sig**2) - sn2 / (4 * sig_s**2)
        c = s2 / (2 * sig**2) + c2 / (2 * sig_s**2)
        g = torch.exp(-(a * dx * dx + 2 * b * dx * dy + c * dy * dy))
        g = g.sum(dim=2) / ACTIVE_AGENTS
        return self.q_obs * g.mean(dim=1) ** 2

    def _static_cost(self, S: torch.Tensor, grid: torch.Tensor | None) -> torch.Tensor:
        if grid is None:
            return torch.zeros(S.shape[0], device=self.device, dtype=self.dtype)
        hits = self.grid.lookup(grid, S[:, :, :2]).to(self.dtype)
        return COLLISION_STATE_COST * hits.sum(dim=1)

    # ---- boundary-aware predictions (node.boundary_aware_predictions) ------------------------
    def _boundary_aware(self, P: np.ndarray, occ: np.ndarray | None) -> np.ndarray:
        if occ is None or P.shape[1] == 0:
            return P
        ox, oy, res, n = self.grid.ox, self.grid.oy, self.grid.res, self.grid.n
        for k in range(2, P.shape[0]):  # upstream index i>0 of the future-only array
            col = np.floor((P[k, :, 0] - ox) / res).astype(np.int64)
            row = np.floor((P[k, :, 1] - oy) / res).astype(np.int64)
            inside = (row >= 0) & (row < n) & (col >= 0) & (col < n)
            hit = np.zeros(P.shape[1], dtype=bool)
            hit[inside] = occ[row[inside], col[inside]]
            P[k, hit] = P[k - 1, hit]
        return P

    # ---- MPPI (pytorch_mppi.MPPI.command, shift_nominal_trajectory=True) -------------------
    def command(
        self,
        pose: np.ndarray,
        goal: np.ndarray,
        preds: np.ndarray,
        obstacle_points: np.ndarray | None,
    ) -> tuple[float, float]:
        dev, dt_ = self.device, self.dtype
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])

        # node.timer_callback turning_to_center: rotate in place towards the goal first.
        if self.turning:
            self.turn_ticks += 1
            delta = math.atan2(goal[1] - y, goal[0] - x) - th
            delta = (delta + math.pi) % (2 * math.pi) - math.pi
            if abs(delta) < TURN_DONE or self.turn_ticks > 100 or math.hypot(goal[0] - x, goal[1] - y) < 0.05:
                self.turning = False
            else:
                w = TURN_W_SLOW if abs(delta) < TURN_SLOW_BAND else TURN_W_FAST
                return 0.0, float(math.copysign(min(w, self.lim.w_max), delta))

        grid_np = occ_np = None
        if obstacle_points is not None:
            grid_np = self.grid.build(x, y, obstacle_points, self.inflation)
            occ_np = self.grid.build(x, y, obstacle_points, 0.1)
        grid = torch.from_numpy(grid_np).to(dev) if grid_np is not None else None
        P = torch.from_numpy(self._boundary_aware(preds.copy(), occ_np)[1:]).to(dev, dt_)  # future only
        g = torch.tensor(goal[:2], device=dev, dtype=dt_)

        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init
        noise = torch.randn(self.K, self.T, 2, generator=self.gen, device=dev, dtype=dt_) * self.noise_std
        perturbed = torch.max(torch.min(self.U + noise, self.u_max), self.u_min)
        noise = perturbed - self.U
        action_cost = LAMBDA * noise @ self.sigma_inv

        s = torch.tensor([x, y, th], device=dev, dtype=dt_).expand(self.K, 3)
        states = []
        for t in range(self.T):
            s = dynamics(s, perturbed[:, t])
            states.append(s)
        S = torch.stack(states, dim=1)  # (K, T, 3)

        cost = self._goal_cost(S, g) + self._static_cost(S, grid)
        if P.shape[1] > 0:
            cost = cost + self._ped_cost(S, P)
        cost = cost + torch.sum(self.U * action_cost, dim=(1, 2))

        beta = torch.min(cost)
        w = torch.exp(-(cost - beta) / LAMBDA)
        w = w / w.sum()
        self.U = self.U + torch.sum(w[:, None, None] * noise, dim=0)
        action = self.U[0].clone()
        self.u_init = action  # node.py: self.mppi.u_init = action
        self.last_states, self.last_cost = S, cost
        v = float(min(max(action[0].item(), 0.0), self.lim.v_max))
        w_ = float(min(max(action[1].item(), -self.lim.w_max), self.lim.w_max))
        return v, w_
