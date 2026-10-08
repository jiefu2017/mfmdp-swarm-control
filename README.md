# mfmdp-swarm-control

Code for chance-constrained reach-avoid control of large swarms modeled as a
mean-field MDP. The method propagates the mean **and covariance** of the
swarm's empirical state distribution, turns per-timestep chance constraints
into deterministic second-order cone constraints (Cantelli bound), and solves
the resulting problem by sequential convex approximation (SCA). It is compared
against a mean-only occupation-measure LP baseline (MF-LP), and both are
evaluated by Monte Carlo simulation of a finite-N swarm.

## Layout

| File | Contents |
|---|---|
| `sim/sca.py` | SCA algorithm (Algorithm 1): Jacobians via `jax.jacrev` through the unrolled mean-field recurrence, one SOCP per iteration via cvxpy |
| `sim/mf_lp.py` | MF-LP baseline: occupation-measure LP enforcing mean constraints only |
| `sim/evaluate.py` | Monte Carlo evaluation of a policy on a finite-N swarm |
| `sim/mfmdp.py` | 4-state stochastic graph MDP and its mean/covariance dynamics |
| `sim/gridworld.py` | 6x6 gridworld with walls and unsafe regions |
| `sim/evcharging.py` | Overnight EV-charging aggregator, with a congestion constraint on the fast-charge occupation measure |
| `sim/run_*_case_study.py` | Scripts that reproduce each case study (MF-LP vs. SCA) |
| `sim/plot_*.py`, `sim/visualize.py`, `sim/make_videos.py` | Figures and swarm-trajectory videos |

All environment modules expose the same interface, so `sca.py`, `mf_lp.py`
and `evaluate.py` run unchanged on any of them.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Rendering videos (`make_videos.py`) also needs [ffmpeg](https://ffmpeg.org/)
on your `PATH`.

## Usage

Run everything from the repository root, as modules:

```bash
python -m sim.run_graph_case_study        # 4-state graph, T = 4
python -m sim.run_gridworld_case_study    # 6x6 gridworld, T = 20
python -m sim.run_evcharging_case_study   # EV charging, T = 12

python -m sim.plot_gridworld_layout       # -> figures/gridworld_layout.pdf
python -m sim.plot_ev_idle_effect         # -> figures/ev_idle_effect.pdf
python -m sim.make_videos                 # -> sim/mf_lp_swarm.mp4, sim/sca_swarm.mp4
```

## License

GPL-3.0. See [LICENSE](LICENSE).
