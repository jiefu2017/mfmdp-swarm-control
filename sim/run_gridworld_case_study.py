"""Runs the 6x6 gridworld case study (acc_paper.tex Sec. VI.B) with the
*corrected* SCA algorithm (merit-function accept/reject, grow-on-
infeasible, LP-based warm start) and per-cell (not aggregate) avoid
constraints on the three congestion gap cells.
"""
import time

from sim import gridworld as env
from sim.mf_lp import solve_mf_lp, recover_policy, lp_warm_start
from sim.sca import sca_algorithm
from sim.evaluate import monte_carlo, mf_lp_policy_fn, sca_policy_fn

T = 20
ALPHA_R = 0.3
DELTA_R = 0.1
BETA_U_CELL, DELTA_U_CELL = 0.30, 0.40  # per-cell: >30% simultaneous occupancy w.p.<0.40
AVOID_REGIONS = [(c, BETA_U_CELL, DELTA_U_CELL) for c in env.C_UNSAFE_CELLS]
FLEET_SIZES = [10, 20, 50, 100]
N_TRIALS = 500
START_STATE = env._idx(*env.START)


def main():
    print("Solving Mean-Field LP baseline ...")
    t0 = time.time()
    lp = solve_mf_lp(env, T, ALPHA_R, AVOID_REGIONS, tstar_range=range(4, T + 1))
    lp_pi = recover_policy(lp["z"])
    lp_fn = mf_lp_policy_fn(lp_pi)
    print(f"  MF-LP: t*={lp['tstar']}, J={lp['J']:.3f}  ({time.time()-t0:.1f}s)")

    theta0, _ = lp_warm_start(env, T, ALPHA_R, AVOID_REGIONS,
                               tstar_range=range(4, T + 1), eps=1e-3)

    print(f"{'N':>4} | {'MF-LP':^28} | {'SCA':^40}")
    print(f"{'':>4} | {'J':>6} {'P_r':>6} {'P_v':>6} {'cert':>4} | "
          f"{'J':>6} {'P_r':>6} {'P_v':>6} {'r_m':>7} {'a_m':>7}")
    print("-" * 90)

    for N in FLEET_SIZES:
        t0 = time.time()
        sca_res = sca_algorithm(env, T, N, ALPHA_R, DELTA_R, AVOID_REGIONS,
                                 K=25, K_tstar=3, rho0=2.0, rho_max=15.0,
                                 theta0=theta0)
        sca_fn = sca_policy_fn(env, sca_res["theta"])

        lp_mc = monte_carlo(env, lp_fn, T, N, ALPHA_R, AVOID_REGIONS, lp["tstar"],
                             n_trials=N_TRIALS, seed=N, start_state=START_STATE)
        sca_mc = monte_carlo(env, sca_fn, T, N, ALPHA_R, AVOID_REGIONS, sca_res["tstar"],
                              n_trials=N_TRIALS, seed=N, start_state=START_STATE)

        # Certification must check each region's OWN violation rate against
        # its OWN delta_u, not an aggregate "did any region violate"
        # statistic against a single threshold (the aggregate is a union
        # over regions and is not directly comparable to any one region's
        # budget).
        lp_cert = "OK" if (lp_mc["P_reach_hat"] >= 1 - DELTA_R and
                            all(p <= DELTA_U_CELL for p in lp_mc["P_viol_per_region"])) else "x"
        sca_cert = "OK" if (sca_mc["P_reach_hat"] >= 1 - DELTA_R and
                             all(p <= DELTA_U_CELL for p in sca_mc["P_viol_per_region"])) else "x"

        print(f"{N:>4} | {lp_mc['J_hat']:>6.2f} {lp_mc['P_reach_hat']:>6.2f} "
              f"{lp_mc['P_viol_hat']:>6.2f} {str(lp_mc['P_viol_per_region'])} {lp_cert:>4} | "
              f"{sca_mc['J_hat']:>6.2f} {sca_mc['P_reach_hat']:>6.2f} "
              f"{sca_mc['P_viol_hat']:>6.2f} {str(sca_mc['P_viol_per_region'])} {sca_cert:>4} "
              f"{sca_res['r_m']:>+7.3f} "
              f"{sca_res['a_m']:>+7.3f}   t*={sca_res['tstar']:2d} "
              f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
