"""Runs the EV-charging aggregator case study: MF-LP baseline vs. SCA
(via automatic differentiation), Monte Carlo validated across a fleet-size
sweep. The avoid constraint here is on the OCCUPATION MEASURE (aggregate
fast-charge action / transformer congestion), not the state marginal --
see evcharging.action_aggregate_sigma (Corollary to Proposition 1).
"""
import time

from sim import evcharging as env
from sim.mf_lp import solve_mf_lp, recover_policy, lp_warm_start
from sim.sca import sca_algorithm
from sim.evaluate import monte_carlo, mf_lp_policy_fn, sca_policy_fn

T = 12
ALPHA_R = 0.8
DELTA_R = 0.1
BETA_U, DELTA_U = 0.5, 0.3  # kappa_u = sqrt(0.7/0.3) ~= 1.53
ACTION_AVOID_REGIONS = [(env.C_FAST_SA, BETA_U, DELTA_U)]
FLEET_SIZES = [20, 50, 100, 200]
N_TRIALS = 500


def main():
    print("Solving MF-LP baseline ...")
    # t* fixed at the terminal time T (not existential over [T]):
    # the reach target (>=80% of the fleet at SOC>=80%) is a departure-time
    # requirement, and the avoid/congestion constraint is enforced over the
    # full charging window t=0,...,T-1.
    lp = solve_mf_lp(env, T, ALPHA_R, avoid_regions=[], tstar_range=[T],
                      action_avoid_regions=ACTION_AVOID_REGIONS)
    lp_pi = recover_policy(lp["z"])
    lp_fn = mf_lp_policy_fn(lp_pi)
    print(f"  MF-LP: t*={lp['tstar']}, J={lp['J']:.3f}")

    # eps=1e-3 (used in the earlier T=10 design) saturates the softmax here:
    # with the reach constraint no longer binding, the LP goes bang-bang on
    # fast/slow at most states (61/73 occupied (t,s) cells within 1e-3 of
    # 0 or 1), killing the policy gradient pi(1-pi) almost everywhere and
    # freezing the SOCP at the warm start. eps=0.1 desaturates it enough to
    # move, but a sweep at N=20,50,100,200 found eps=0.2 reaches full
    # certification (a_m~=-0.000) from N=50 upward, vs. only N=200 at
    # eps=0.1 -- eps=0.2 is used throughout for a fair, non-per-N-tuned
    # comparison.
    theta0, _ = lp_warm_start(env, T, ALPHA_R, avoid_regions=[],
                               tstar_range=[lp["tstar"]],
                               action_avoid_regions=ACTION_AVOID_REGIONS, eps=0.2)

    print(f"{'N':>4} | {'MF-LP':^20} | {'SCA':^40}")
    print(f"{'':>4} | {'J':>6} {'P_r':>6} {'P_v':>6} | "
          f"{'J':>6} {'P_r':>6} {'P_v':>6} {'r_m':>7} {'a_m':>7}")
    print("-" * 82)

    for N in FLEET_SIZES:
        t0 = time.time()
        sca_res = sca_algorithm(env, T, N, ALPHA_R, DELTA_R, avoid_regions=[],
                                 action_avoid_regions=ACTION_AVOID_REGIONS,
                                 tstar_candidates=[lp["tstar"]],
                                 K=25, rho0=2.0, rho_max=15.0, theta0=theta0)
        sca_fn = sca_policy_fn(env, sca_res["theta"])

        lp_mc = monte_carlo(env, lp_fn, T, N, ALPHA_R, avoid_regions=[],
                             tstar=lp["tstar"], n_trials=N_TRIALS, seed=N,
                             action_avoid_regions=ACTION_AVOID_REGIONS)
        sca_mc = monte_carlo(env, sca_fn, T, N, ALPHA_R, avoid_regions=[],
                              tstar=sca_res["tstar"], n_trials=N_TRIALS, seed=N,
                              action_avoid_regions=ACTION_AVOID_REGIONS)

        print(f"{N:>4} | {lp_mc['J_hat']:>6.2f} {lp_mc['P_reach_hat']:>6.2f} "
              f"{lp_mc['P_viol_per_region'][0]:>6.2f} | "
              f"{sca_mc['J_hat']:>6.2f} {sca_mc['P_reach_hat']:>6.2f} "
              f"{sca_mc['P_viol_per_region'][0]:>6.2f} {sca_res['r_m']:>+7.3f} "
              f"{sca_res['a_m']:>+7.3f}   t*={sca_res['tstar']:2d} "
              f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
