# hmpnav

Arena wrapper for **HMPNav**, MPPI over a unicycle model with constant-velocity human motion prediction. Adapted from [fluentrobotics/HMPNav](https://github.com/fluentrobotics/HMPNav) (commit 8b23db2), CV predictor only.

## Run

```sh
arena launch robot.mobile:=drl robot.mobile.planner:=hmpnav
```

Requires a global plan. Defaults to `nav2/navfn`. Velocity limits and radius are the Jackal's.

## Files

- `planner.py`: entry point. Robot pose, plan subgoal, tracked pedestrians and the laser scan produce `[v, omega]`. Its docstring lists the `HMPNAV_*` environment knobs (profile, samples, device, threads).
- `hmpnav_core.py`: ROS-free port of the upstream MPPI controller and CV predictor, with the deviations for Arena listed in its docstring.
- `offline_test.py`: ROS-free crossing scenarios against synthetic pedestrians.
- `planner.yaml`: observation manifest.

## License

BSD-3-Clause (inherited from upstream, see [LICENSE.HMPNav](LICENSE.HMPNav)).
