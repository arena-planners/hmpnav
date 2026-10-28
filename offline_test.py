"""Offline harness for the hmpnav planner: synthetic features, no ROS, no bridge.

Robot at (30, 40) facing +x, goal (44, 40). The goal_pose fed to step() is a 2 m lookahead on the
straight-line plan (what SubgoalGenerator produces in arena). Pedestrians walk straight lines at
constant speed and ignore the robot; they are timed to cross the robot's straight path when a
0.5 m/s robot would be there. The laser scan is ray-cast (640 beams, 12 m, canonical layout)
against walls and pedestrian discs. The robot is a unicycle with Jackal accel limits.

Usage: CUDA_VISIBLE_DEVICES= python offline_test.py [max_steps] [n_seeds]
(n_seeds > 1 repeats the crossing scenarios with different pedestrian layouts and prints a summary)
"""

from __future__ import annotations

import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import planner  # noqa: E402

DT = 0.1
ROBOT_R = 0.267
PED_R = 0.3
BEAMS, RANGE = 640, 12.0
START, GOAL = np.array([30.0, 40.0, 0.0]), np.array([44.0, 40.0])


def crossing_peds(n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    peds = []
    xs = np.linspace(33.0, 42.5, n) if n else []
    for i, x in enumerate(xs):
        speed = rng.uniform(0.6, 1.2)
        t_arrive = (x - START[0]) / 0.5 + rng.uniform(-1.0, 1.0)
        side = 1.0 if i % 2 == 0 else -1.0
        p0 = np.array([x + rng.uniform(-0.5, 0.5), 40.0 - side * speed * t_arrive])
        peds.append((p0, np.array([0.0, side * speed])))
    return peds


def ped_positions(peds, t):
    return np.array([p0 + v * t for p0, v in peds]).reshape(-1, 2)


def raycast(pose, walls, ped_xy):
    ang = pose[2] + 2 * np.pi * np.arange(BEAMS) / BEAMS
    d = np.stack([np.cos(ang), np.sin(ang)], 1)
    o = pose[:2]
    r = np.full(BEAMS, RANGE)
    for (a, b) in walls:
        a, b = np.asarray(a, float), np.asarray(b, float)
        e = b - a
        den = d[:, 0] * (-e[1]) - d[:, 1] * (-e[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            w = a - o
            t = (w[0] * (-e[1]) - w[1] * (-e[0])) / den
            u = (d[:, 0] * w[1] - d[:, 1] * w[0]) / den
        hit = (np.abs(den) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
        r = np.where(hit, np.minimum(r, t), r)
    for c in ped_xy:
        oc = o - c
        bq = d @ oc
        disc = bq * bq - (oc @ oc - PED_R**2)
        t = -bq - np.sqrt(np.maximum(disc, 0))
        r = np.where((disc > 0) & (t > 0), np.minimum(r, t), r)
    return r.astype(np.float32)


def wall_clearance(p, walls):
    best = np.inf
    for a, b in walls:
        a, b = np.asarray(a, float), np.asarray(b, float)
        e = b - a
        t = np.clip((p - a) @ e / (e @ e), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * e))))
    return best


def run(name, peds, walls, max_steps):
    planner._runner = None
    planner.on_reset(name, None)
    pose = START.copy()
    v = w = 0.0
    min_ped = np.inf
    min_wall = np.inf
    times = []
    reached = None
    dist100 = float("nan")
    for k in range(max_steps):
        t = k * DT
        pxy = ped_positions(peds, t)
        vel = np.array([pv for _, pv in peds]).reshape(-1, 2)
        feats = {
            "robot_pose": pose.astype(np.float32),
            "robot_state": np.array([pose[0], pose[1], v, 0.0, pose[2], w], np.float32),
            "goal_pose": None,
            "pedestrians": np.column_stack([np.arange(len(peds)), pxy, vel]).astype(np.float32)
            if len(peds)
            else np.empty((0, 5), np.float32),
            "laser_scan": raycast(pose, walls, pxy),
        }
        sub_x = min(pose[0] + 2.0, GOAL[0]) if np.linalg.norm(GOAL - pose[:2]) > 2.0 else GOAL[0]
        feats["goal_pose"] = np.array([sub_x, GOAL[1], 0.0], np.float32)
        t0 = time.perf_counter()
        cmd_v, cmd_w = planner.step(feats)
        times.append(time.perf_counter() - t0)
        # Jackal: |v|<=0.5, |w|<=1.5, accel 2.5 m/s^2, 3.2 rad/s^2
        cmd_v = float(np.clip(cmd_v, -0.5, 0.5))
        cmd_w = float(np.clip(cmd_w, -1.5, 1.5))
        v += float(np.clip(cmd_v - v, -2.5 * DT, 2.5 * DT))
        w += float(np.clip(cmd_w - w, -3.2 * DT, 3.2 * DT))
        pose[0] += v * math.cos(pose[2]) * DT
        pose[1] += v * math.sin(pose[2]) * DT
        pose[2] = (pose[2] + w * DT + math.pi) % (2 * math.pi) - math.pi
        if len(peds):
            min_ped = min(min_ped, float(np.min(np.linalg.norm(ped_positions(peds, t + DT) - pose[:2], axis=1))))
        if walls:
            min_wall = min(min_wall, wall_clearance(pose[:2], walls))
        if k == 99:
            dist100 = float(np.linalg.norm(GOAL - pose[:2]))
        if np.linalg.norm(GOAL - pose[:2]) < 0.3:
            reached = k + 1
            break
    ms = 1e3 * np.array(times[1:])
    coll_p = min_ped < ROBOT_R + PED_R
    coll_w = min_wall < ROBOT_R
    print(
        f"{name:22s} reached={'yes @%3d steps (%.1fs)' % (reached, reached * DT) if reached else 'NO ':22s} "
        f"goal_dist@100={dist100:5.2f}m final=({pose[0]:.2f},{pose[1]:.2f}) "
        f"min_ped={min_ped:5.2f}m{' COLLISION' if coll_p else ''} "
        f"min_wall={min_wall:5.2f}m{' COLLISION' if coll_w else ''} "
        f"ms/step mean={ms.mean():.1f} p95={np.percentile(ms, 95):.1f} max={ms.max():.1f}"
    )
    return reached, min_ped, min_wall, ms


def main():
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    corridor = [((28, 38.5), (46, 38.5)), ((28, 41.5), (46, 41.5))]
    head_on = [(np.array([45.0, 40.3]), np.array([-0.9, 0.0])), (np.array([48.0, 39.6]), np.array([-1.1, 0.0]))]
    pillar = [((36.5, 39.4), (37.5, 39.4)), ((37.5, 39.4), (37.5, 40.6)), ((37.5, 40.6), (36.5, 40.6)), ((36.5, 40.6), (36.5, 39.4))]
    scen = [
        ("open_0peds", [], []),
        ("corridor_2peds_headon", head_on, corridor),
        ("pillar_on_path_0peds", [], pillar),
    ]
    for seed in range(1, n_seeds + 1):
        scen.append((f"open_3peds_crossing_s{seed}", crossing_peds(3, seed), []))
        scen.append((f"open_8peds_crossing_s{seed}", crossing_peds(8, 100 + seed), []))
    res = [run(name, peds, walls, max_steps) for name, peds, walls in scen]
    ms = np.concatenate([r[3] for r in res])
    n_goal = sum(r[0] is not None for r in res)
    n_coll = sum(r[1] < ROBOT_R + PED_R or r[2] < ROBOT_R for r in res)
    print(
        f"SUMMARY runs={len(res)} reached={n_goal} collisions={n_coll} "
        f"min_ped_median={np.median([r[1] for r in res if np.isfinite(r[1])]):.2f}m "
        f"ms/step mean={ms.mean():.1f} p95={np.percentile(ms, 95):.1f}"
    )


if __name__ == "__main__":
    main()
