"""Driver script: computes the MF-LP and SCA policies (final gridworld
design) and renders one video each via sim/visualize.py.
"""
from sim import gridworld as env
from sim.mf_lp import solve_mf_lp, recover_policy, lp_warm_start
from sim.sca import sca_algorithm
from sim.evaluate import mf_lp_policy_fn, sca_policy_fn
from sim.visualize import make_video

T = 20
ALPHA_R = 0.3
DELTA_R = 0.1
AVOID_REGIONS = [(c, 0.30, 0.40) for c in env.C_UNSAFE_CELLS]
N = 50

if __name__ == "__main__":
    print("Solving MF-LP ...")
    lp = solve_mf_lp(env, T, ALPHA_R, AVOID_REGIONS, tstar_range=range(4, T + 1))
    lp_pi = recover_policy(lp["z"])
    lp_fn = mf_lp_policy_fn(lp_pi)
    print(f"  MF-LP: t*={lp['tstar']}, J={lp['J']:.3f}")

    print("Solving SCA ...")
    theta0, _ = lp_warm_start(env, T, ALPHA_R, AVOID_REGIONS,
                               tstar_range=range(4, T + 1), eps=1e-3)
    sca_res = sca_algorithm(env, T, N, ALPHA_R, DELTA_R, AVOID_REGIONS,
                             K=25, K_tstar=3, rho0=2.0, rho_max=15.0, theta0=theta0)
    sca_fn = sca_policy_fn(env, sca_res["theta"])
    print(f"  SCA: t*={sca_res['tstar']}, J={sca_res['J']:.3f}, "
          f"r_m={sca_res['r_m']:+.3f}, a_m={sca_res['a_m']:+.3f}")

    print("Rendering MF-LP video ...")
    make_video(lp_fn, T, N, ALPHA_R, AVOID_REGIONS, lp["tstar"],
               "sim/mf_lp_swarm.mp4", "MF-LP policy (N=50)", seed=1)

    print("Rendering SCA video ...")
    make_video(sca_fn, T, N, ALPHA_R, AVOID_REGIONS, sca_res["tstar"],
               "sim/sca_swarm.mp4", "SCA policy (N=50)", seed=1)
