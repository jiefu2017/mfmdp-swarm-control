"""Standalone visualization: renders swarm-trajectory videos for the MF-LP
and SCA policies on the gridworld, each with a side panel showing
chance-constraint satisfaction over time.

Kept separate from sim/mfmdp.py, sim/sca.py, sim/mf_lp.py, sim/evaluate.py
(environment/algorithm code) since this is purely for visualization -- it
imports from those modules but adds no modeling logic of its own.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle

from sim import gridworld as env
from sim.evaluate import _sample_actions


def simulate_single_trial(policy_fn, T, num_agents, start_state, seed=0):
    """Simulates ONE trial (not an aggregated Monte Carlo average), tracking
    per-agent positions and per-timestep density statistics for animation.
    """
    rng = np.random.default_rng(seed)
    P_np = np.array(env.P)
    c_target = np.array(env.C_TARGET)
    c_cells = [np.array(c) for c in env.C_UNSAFE_CELLS]

    states = np.full(num_agents, start_state, dtype=int)
    states_hist = [states.copy()]
    mu0 = np.bincount(states, minlength=env.S) / num_agents
    mu_target_hist = [float(mu0 @ c_target)]
    mu_cells_hist = [[float(mu0 @ c) for c in c_cells]]

    for t in range(T):
        mu_hat = np.bincount(states, minlength=env.S) / num_agents
        pi_t = np.array(policy_fn(t, mu_hat))
        actions = _sample_actions(rng, pi_t[states])
        next_states = np.empty_like(states)
        for s in np.unique(states):
            idx = np.where(states == s)[0]
            for a in np.unique(actions[idx]):
                sub = idx[actions[idx] == a]
                next_states[sub] = rng.choice(env.S, size=sub.size, p=P_np[s, a])
        states = next_states
        states_hist.append(states.copy())
        mu_next = np.bincount(states, minlength=env.S) / num_agents
        mu_target_hist.append(float(mu_next @ c_target))
        mu_cells_hist.append([float(mu_next @ c) for c in c_cells])

    return states_hist, mu_target_hist, mu_cells_hist


def _draw_static_grid(ax):
    ax.set_xlim(-0.5, env.GRID - 0.5)
    ax.set_ylim(-0.5, env.GRID - 0.5)
    ax.set_xticks(range(env.GRID))
    ax.set_yticks(range(env.GRID))
    ax.grid(True, color="lightgray", linewidth=0.5)
    ax.set_aspect("equal")

    for (x, y) in env.WALL_CELLS:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="black"))
    for (x, y) in env.UNSAFE_CELLS:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="#ffcc66", alpha=0.7))
    for (x, y) in env.REACH_TARGETS:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="#66cc66", alpha=0.7))
    rx, ry = env.REWARD_ONLY_TARGET
    ax.add_patch(Rectangle((rx - 0.5, ry - 0.5), 1, 1, color="#6699ff", alpha=0.7))
    sx, sy = env.START
    ax.add_patch(Rectangle((sx - 0.5, sy - 0.5), 1, 1, fill=False,
                            edgecolor="red", linewidth=2))


def make_video(policy_fn, T, num_agents, alpha_r, avoid_regions, tstar,
               out_path, title, seed=0, fps=2):
    """Renders a two-panel video: swarm positions on the grid (left) and
    reach/avoid density vs. thresholds over time (right).
    """
    states_hist, mu_target_hist, mu_cells_hist = simulate_single_trial(
        policy_fn, T, num_agents, env._idx(*env.START), seed=seed)
    beta_u = avoid_regions[0][1]

    fig, (ax_grid, ax_chart) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(title)

    _draw_static_grid(ax_grid)
    ax_grid.set_title("Swarm positions")
    scat = ax_grid.scatter([], [], s=18, color="crimson", alpha=0.7, zorder=5)

    ax_chart.set_xlim(0, T)
    ax_chart.set_ylim(0, 1)
    ax_chart.set_xlabel("$t$")
    ax_chart.set_ylabel("density")
    ax_chart.axhline(alpha_r, color="green", linestyle="--", linewidth=1,
                      label=r"$\alpha_r$")
    ax_chart.axhline(beta_u, color="orange", linestyle="--", linewidth=1,
                      label=r"$\beta_u$")
    ax_chart.axvline(tstar, color="gray", linestyle=":", linewidth=1,
                      label=r"$t^*$")
    line_target, = ax_chart.plot([], [], color="green",
                                  label=r"$\bar\mu_t(\mathcal{T})$")
    line_unsafe, = ax_chart.plot([], [], color="orange",
                                  label=r"$\max_i\bar\mu_t(s_i)$")
    ax_chart.legend(loc="upper right", fontsize=7)
    ax_chart.set_title("Chance-constraint satisfaction")

    def update(frame):
        pts = np.array([env._xy(s) for s in states_hist[frame]], dtype=float)
        jitter = (np.random.default_rng(1000 + frame).random((len(pts), 2)) - 0.5) * 0.7
        scat.set_offsets(pts + jitter)

        ts = list(range(frame + 1))
        line_target.set_data(ts, mu_target_hist[:frame + 1])
        max_unsafe = [max(c) for c in mu_cells_hist[:frame + 1]]
        line_unsafe.set_data(ts, max_unsafe)
        ax_grid.set_xlabel(f"$t={frame}$")
        return scat, line_target, line_unsafe

    anim = FuncAnimation(fig, update, frames=T + 1, blit=False)
    anim.save(out_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)
    print(f"Saved {out_path}")
