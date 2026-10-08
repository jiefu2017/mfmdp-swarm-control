"""Runs the 4-state stochastic graph case study (acc_paper.tex Sec. VI.A):
MF-LP baseline vs. SCA (via automatic differentiation), Monte Carlo
validated across a fleet-size sweep. Produces a table analogous to
Table I.
"""
import time

from sim import mfmdp as env
from sim.mf_lp import solve_mf_lp, recover_policy
from sim.sca import sca_algorithm
from sim.evaluate import monte_carlo, mf_lp_policy_fn, sca_policy_fn

T = 4
ALPHA_R, BETA_U = 0.40, 0.55
DELTA_R, DELTA_U = 0.10, 0.10
AVOID_REGIONS = [(env.C_UNSAFE, BETA_U, DELTA_U)]
FLEET_SIZES = [10, 20, 50, 100]
N_TRIALS = 2000


def main():
    print("Solving Mean-Field LP baseline ...")
    lp = solve_mf_lp(env, T, ALPHA_R, AVOID_REGIONS)
    lp_pi = recover_policy(lp["z"])
    lp_fn = mf_lp_policy_fn(lp_pi)
    print(f"  MF-LP: t*={lp['tstar']}, J={lp['J']:.3f}")

    print(f"{'N':>4} | {'MF-LP':^28} | {'SCA':^40}")
    print(f"{'':>4} | {'J':>6} {'P_r':>6} {'P_v':>6} {'cert':>4} | "
          f"{'J':>6} {'P_r':>6} {'P_v':>6} {'r_m':>7} {'a_m':>7}")
    print("-" * 90)

    for N in FLEET_SIZES:
        t0 = time.time()
        sca_res = sca_algorithm(env, T, N, ALPHA_R, DELTA_R, AVOID_REGIONS,
                                 K=30, K_tstar=3, rho0=2.0, rho_max=15.0)
        sca_fn = sca_policy_fn(env, sca_res["theta"])

        lp_mc = monte_carlo(env, lp_fn, T, N, ALPHA_R, AVOID_REGIONS, lp["tstar"],
                             n_trials=N_TRIALS, seed=N)
        sca_mc = monte_carlo(env, sca_fn, T, N, ALPHA_R, AVOID_REGIONS, sca_res["tstar"],
                              n_trials=N_TRIALS, seed=N)

        lp_cert = "OK" if (lp_mc["P_reach_hat"] >= 0.90 and
                            lp_mc["P_viol_hat"] <= 0.10) else "x"

        print(f"{N:>4} | {lp_mc['J_hat']:>6.2f} {lp_mc['P_reach_hat']:>6.2f} "
              f"{lp_mc['P_viol_hat']:>6.2f} {lp_cert:>4} | "
              f"{sca_mc['J_hat']:>6.2f} {sca_mc['P_reach_hat']:>6.2f} "
              f"{sca_mc['P_viol_hat']:>6.2f} {sca_res['r_m']:>+7.3f} "
              f"{sca_res['a_m']:>+7.3f}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
