"""Illustrates Sec. VI.E (acc_paper.tex): compares how the MF-LP
baseline and the SCA policy each use the fast-charging vs. idle
actions over the 12-timestep window. MF-LP (no variance correction)
saturates the beta_u=0.5 congestion cap at every timestep, which is
exactly why its realised violation rate sits at the CLT ~50% floor
(Key Finding (1)); SCA leaves a margin below the cap, which is why it
certifies (Key Finding (2)). Separate from the algorithm/case-study
code, per the project's convention of keeping visualization code in
its own module.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

from sim import evcharging as env
from sim.mf_lp import solve_mf_lp, lp_warm_start
from sim.sca import sca_algorithm

T = 12
ALPHA_R = 0.8
BETA_U, DELTA_U = 0.5, 0.3
ACTION_AVOID_REGIONS = [(env.C_FAST_SA, BETA_U, DELTA_U)]
N_SCA = 100  # a fleet size at which SCA is fully certified (a_m ~= -0.000)


def _fractions_per_t(z):
    """z: (T, S, A). Returns (fast_frac, slow_frac, idle_frac) per timestep,
    summing to 1 (the whole fleet) at every t."""
    fast = np.array([float(np.sum(z[t, :, env.FAST])) for t in range(z.shape[0])])
    slow = np.array([float(np.sum(z[t, :, env.SLOW])) for t in range(z.shape[0])])
    idle = np.array([float(np.sum(z[t, :, env.IDLE])) for t in range(z.shape[0])])
    return fast, slow, idle


def _panel(ax, fast, slow, idle, title):
    ts = np.arange(len(fast))
    ax.axhline(BETA_U, color="gray", linestyle=":", linewidth=1.2,
               label=r"cap $\beta_u=0.5$")
    ax.plot(ts, fast, "o-", color="#3366cc", markersize=4, label="fast-charging fraction")
    ax.plot(ts, slow, "^-", color="#cc8833", markersize=4, label="slow-charging fraction")
    ax.plot(ts, idle, "s-", color="#66aa66", markersize=4, label="idle fraction")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("fraction of fleet")
    ax.set_ylim(-0.02, 0.62)
    ax.set_xticks(ts)


def main(out_path="figures/ev_idle_effect.pdf"):
    lp = solve_mf_lp(env, T, ALPHA_R, avoid_regions=[], tstar_range=[T],
                      action_avoid_regions=ACTION_AVOID_REGIONS)
    print(f"MF-LP: J={lp['J']:.3f}")

    theta0, _ = lp_warm_start(env, T, ALPHA_R, avoid_regions=[],
                               tstar_range=[lp["tstar"]],
                               action_avoid_regions=ACTION_AVOID_REGIONS, eps=0.2)
    sca_res = sca_algorithm(env, T, N_SCA, ALPHA_R, 0.1, avoid_regions=[],
                             action_avoid_regions=ACTION_AVOID_REGIONS,
                             tstar_candidates=[lp["tstar"]],
                             K=25, rho0=2.0, rho_max=15.0, theta0=theta0)
    print(f"SCA (N={N_SCA}): J={sca_res['J']:.3f} r_m={sca_res['r_m']:+.3f} "
          f"a_m={sca_res['a_m']:+.3f}")

    lp_fast, lp_slow, lp_idle = _fractions_per_t(np.array(lp["z"]))
    sca_fast, sca_slow, sca_idle = _fractions_per_t(np.array(sca_res["zs"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.2, 4.8), sharex=True)
    _panel(ax1, lp_fast, lp_slow, lp_idle, "MF-LP (baseline)")
    _panel(ax2, sca_fast, sca_slow, sca_idle, f"SCA, $N={N_SCA}$ (proposed)")
    ax2.set_xlabel("timestep $t$")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4,
               columnspacing=0.8, handletextpad=0.4,
               fontsize=6.3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
