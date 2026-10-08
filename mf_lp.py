"""Mean-Field LP baseline: occupation-measure LP enforcing the mean
constraints mu_t(T) >= alpha_r, mu_t(U) <= beta_u only (no variance
correction), as described in acc_paper.tex Sec. VI.C.

Generic over any environment module exposing the mfmdp.py interface.
avoid_regions is a list of (c_vec, beta, delta) triplets, one per
independent avoid region; the LP only uses (c_vec, beta) since it is
mean-only (delta/kappa is ignored here, but kept in the triplet so the
same avoid_regions list can be passed to sca.py unchanged).
"""
import numpy as np
import cvxpy as cp
import jax.numpy as jnp


def _solve_for_tstar(env, T, alpha_r, avoid_regions, tstar,
                      action_avoid_regions=()):
    P_flat = np.array(env.P).reshape(env.S * env.A, env.S)
    R_flat = np.array(env.R).reshape(env.S * env.A)
    sel = np.zeros((env.S, env.S * env.A))
    for s in range(env.S):
        for a in range(env.A):
            sel[s, s * env.A + a] = 1.0

    mu0_np = np.array(env.MU0)
    z = [cp.Variable(env.S * env.A, nonneg=True) for _ in range(T)]

    mu = [mu0_np]
    constraints = []
    for t in range(T):
        mu_next = P_flat.T @ z[t]
        constraints.append(sel @ z[t] == mu[t])
        mu.append(mu_next)

    for c_vec, beta, _delta in avoid_regions:
        c_np = np.array(c_vec)
        for t in range(1, tstar + 1):  # mu[0] is fixed data, not a cvxpy expression
            constraints.append(c_np @ mu[t] <= beta)
    for c_sa, beta, _delta in action_avoid_regions:
        c_np = np.array(c_sa)
        # z[t] indexes t=0..T-1; matches sca.py's range(tstar+1) guarded by t<T.
        for t in range(min(tstar + 1, T)):
            constraints.append(c_np @ z[t] <= beta)
    constraints.append(np.array(env.C_TARGET) @ mu[tstar] >= alpha_r)

    objective = cp.Maximize(sum(R_flat @ z[t] for t in range(T)))
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return None
    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None

    z_val = np.stack([z[t].value.reshape(env.S, env.A) for t in range(T)])
    mu_val = np.stack([mu0_np] + [np.array(mu[t + 1].value) for t in range(T)])
    return dict(z=z_val, mu=mu_val, J=float(problem.value), tstar=tstar)


def solve_mf_lp(env, T, alpha_r, avoid_regions, tstar_range=None,
                 action_avoid_regions=()):
    """Solves the MF-LP for each candidate t* and keeps the best."""
    best = None
    for tstar in (tstar_range or range(1, T + 1)):
        res = _solve_for_tstar(env, T, alpha_r, avoid_regions, tstar,
                                action_avoid_regions)
        if res is None:
            continue
        if best is None or res["J"] > best["J"]:
            best = res
    return best


def recover_policy(z_val):
    """pi_t(a|s) = z_t(s,a) / sum_a' z_t(s,a'); uniform fallback if mu_t(s)=0."""
    T, Sd, Ad = z_val.shape
    pi = np.zeros_like(z_val)
    for t in range(T):
        row_sums = z_val[t].sum(axis=1, keepdims=True)
        safe = row_sums.copy()
        safe[safe == 0] = 1.0
        pi[t] = np.where(row_sums > 0, z_val[t] / safe, 1.0 / Ad)
    return pi


def lp_warm_start(env, T, alpha_r, avoid_regions, tstar_range=None, eps=1e-3,
                   action_avoid_regions=()):
    """Warm-starts theta0 to exactly reproduce the MF-LP's recovered policy,
    via the constant bias feature (phi[0]=1 at every state): setting
    theta[t,s,a,0] = log(pi_LP(a|s)) makes softmax(theta[t,s]@phi) = pi_LP
    exactly, regardless of the other (zeroed) feature weights.

    This gives SCA a starting point already near the LP's efficient,
    mean-constraint-satisfying trajectory, instead of a crude geometric
    heuristic SCA then has to discover a good routing from scratch via
    local linearized steps alone.
    """
    lp = solve_mf_lp(env, T, alpha_r, avoid_regions, tstar_range=tstar_range,
                      action_avoid_regions=action_avoid_regions)
    if lp is None:
        raise RuntimeError("MF-LP infeasible; cannot build a warm start from it")
    pi_lp = recover_policy(lp["z"])  # (T, S, A)
    theta0 = np.zeros((T, env.S, env.A, env.N_FEAT))
    theta0[:, :, :, 0] = np.log(pi_lp + eps)
    return jnp.array(theta0), lp
