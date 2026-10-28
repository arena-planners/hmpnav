"""HMPNav (fluentrobotics/HMPNav) MPPI + human motion prediction, wrapped for the arena_planners bridge.

Upstream is a ROS2 Humble node for a Hello Robot Stretch with mocap humans. This adapter keeps its
controller (pytorch_mppi-style MPPI over an exact unicycle arc model, DT 0.4 s, 11-step horizon,
2000 samples, terminal goal + asymmetric-Gaussian pedestrian + static collision cost, rotate-to-goal
start) and its constant-velocity (CV) predictor, and feeds them from arena features:

  robot_pose   [x, y, theta] world (tf)                 -> MPPI state
  goal_pose    subgoal on the global plan (lookahead 2) -> MPPI goal
  pedestrians  [[id, x, y, vx, vy], ...] world          -> per-id history -> CV predictions
  laser_scan   canonical scan (ray 0 = heading, CCW)    -> inflated local grid (static cost)

Only METHOD=cv is available (the paper's CV baseline). HST needs RGB-D skeletons and CoHAN 2.0 a
separate ROS stack, neither exists in arena.

Env knobs (all optional): HMPNAV_PROFILE (arena | upstream, see _PROFILES), HMPNAV_SAMPLES (2000), HMPNAV_DEVICE (cpu), HMPNAV_THREADS (2),
HMPNAV_RATE_HZ (manifest rate_hz; set it if robot.mobile.rate overrides the tick),
HMPNAV_NO_SCAN=1 (drop the laser static cost, upstream-with-no-map behaviour).
"""

from __future__ import annotations

import os
import pathlib
import zlib

import numpy as np
import torch
import yaml

from hmpnav_core import CVPredictor, HMPNavMPPI, Limits

_HERE = pathlib.Path(__file__).parent

# Jackal caps/mobile.yaml: velocity_limits linear [-0.5, 0.5], angular [-1.5, 1.5], radius 0.267.
_V_MAX = 0.5
_W_MAX = 1.5
_ROBOT_RADIUS = 0.267
_PED_SCAN_EXCLUDE = 0.45  # scan returns this close to a tracked pedestrian belong to that person
_SCAN_KEEP = 180  # beams kept for the static grid (every n-th of the canonical scan)

# (Q_obs scale, sigma scale) on the pedestrian cost. "upstream" is HMPNav's controller_config.py
# verbatim (sigma 0.3/0.3/0.6 m, Q_OBS 1e4); with a 0.5 m/s Jackal and non-reactive pedestrians it
# barely deviates and touches people in most offline crossing runs. "arena" (default) widens the
# Gaussians to the Jackal + pedestrian contact distance (~0.57 m) and restores the obstacle/goal
# balance at the higher speed. Individual knobs: HMPNAV_Q_OBS_SCALE, HMPNAV_SIGMA_SCALE.
_PROFILES = {"arena": ("10.0", "2.0"), "upstream": ("1.0", "1.0")}


def _manifest_rate() -> float:
    try:
        with open(_HERE / "planner.yaml") as fh:
            return float((yaml.safe_load(fh) or {}).get("rate_hz", 10.0))
    except OSError:
        return 10.0


class _Runner:
    def __init__(self) -> None:
        torch.set_num_threads(int(os.environ.get("HMPNAV_THREADS", "2")))
        profile = _PROFILES[os.environ.get("HMPNAV_PROFILE", "arena")]
        device = os.environ.get("HMPNAV_DEVICE", "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.mppi = HMPNavMPPI(
            num_samples=int(os.environ.get("HMPNAV_SAMPLES", "2000")),
            limits=Limits(v_max=_V_MAX, w_max=_W_MAX),
            device=device,
            robot_radius=_ROBOT_RADIUS,
            q_obs_scale=float(os.environ.get("HMPNAV_Q_OBS_SCALE", profile[0])),
            sigma_scale=float(os.environ.get("HMPNAV_SIGMA_SCALE", profile[1])),
        )
        rate = float(os.environ.get("HMPNAV_RATE_HZ", _manifest_rate()))
        self.cv = CVPredictor(tick_dt=1.0 / rate)
        self.use_scan = os.environ.get("HMPNAV_NO_SCAN", "0") != "1"

    def reset(self, episode_id: str | None) -> None:
        seed = zlib.crc32(str(episode_id).encode()) if episode_id is not None else 0
        self.mppi.reset(seed)
        self.cv.reset()


_runner: _Runner | None = None


def _get_runner() -> _Runner:
    global _runner
    if _runner is None:
        _runner = _Runner()
    return _runner


def _peds_array(peds) -> np.ndarray:
    if peds is None:
        return np.zeros((0, 5))
    arr = np.asarray(peds, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return np.zeros((0, 5))
    return arr


def _scan_points(scan, pose: np.ndarray, peds: np.ndarray, reach: float) -> np.ndarray | None:
    if scan is None:
        return None
    r = np.asarray(scan, dtype=np.float64).ravel()
    n = r.size
    if n == 0:
        return None
    r_max = float(np.max(r))
    idx = np.arange(0, n, max(1, n // _SCAN_KEEP))
    rr = r[idx]
    ang = pose[2] + 2.0 * np.pi * idx / n
    keep = (rr < r_max - 1e-3) & (rr < reach)
    rr, ang = rr[keep], ang[keep]
    pts = np.stack([pose[0] + rr * np.cos(ang), pose[1] + rr * np.sin(ang)], axis=1)
    if peds.shape[0] and pts.shape[0]:
        d = np.linalg.norm(pts[:, None, :] - peds[None, :, 1:3], axis=2)
        pts = pts[d.min(axis=1) > _PED_SCAN_EXCLUDE]
    return pts


def step(features: dict) -> list[float]:
    runner = _get_runner()
    robot_pose = features.get("robot_pose")
    goal_pose = features.get("goal_pose")
    if robot_pose is None or goal_pose is None:
        return [0.0, 0.0]
    pose = np.asarray(robot_pose, dtype=np.float64)[:3]
    goal = np.asarray(goal_pose, dtype=np.float64)[:2]

    peds = _peds_array(features.get("pedestrians"))
    preds = runner.cv.update(peds)  # (13, H, 2), index 0 = now
    # Pedestrians that cannot reach the rollouts inside the horizon only cost time.
    if peds.shape[0]:
        near = np.linalg.norm(preds - pose[None, None, :2], axis=2).min(axis=0) < 5.0
        preds = preds[:, near]

    pts = None
    if runner.use_scan:
        pts = _scan_points(features.get("laser_scan"), pose, peds, runner.mppi.grid.half * 1.42)
    v, w = runner.mppi.command(pose, goal, preds, pts)
    return [v, w]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    _get_runner().reset(episode_id)


if __name__ == "__main__":
    from arena_planners.sdk import load_manifest, main_loop

    main_loop(step, manifest=load_manifest(_HERE / "planner.yaml"), on_reset=on_reset)
