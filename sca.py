"""Algorithm 1 (SCA for Reach-Avoid Swarm Control) from acc_paper.tex,
using automatic differentiation (jax.jacrev through the unrolled
mean-field recurrence) for G_t^(k), D_t^(k) instead of hand-coding the
Proposition 3 recursion, and cvxpy for the per-iteration SOCP.

Generic over any environment module exposing the mfmdp.py interface
(S, A, N_FEAT, P, R, MU0, C_TARGET, forward_sim, lyapunov_recursion,
cantelli_sigma, kappa, total_reward), so the same algorithm runs on the
4-state graph (mfmdp.py) or the gridworld (gridworld.py).

Avoid constraints are a *list* of independent regions, each with its own
indicator vector, density threshold, and reliability level -- e.g. three
separate per-cell congestion constraints rather than one aggregate
constraint on their sum. Each region is specified as (c_vec, beta, delta).
"""
import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp


def jacobians(env, theta):
    """Returns D_all (T+1,S,n_theta), G_all (T,S*A,n_theta) via reverse-mode AD."""
    T, Sd, Ad, F = theta.shape
    n_theta = T * Sd * Ad * F

    def mus_zs(th):
        mus, zs, _ = env.forward_sim(th)
        return mus, zs

    jac_mus, jac_zs = jax.jacrev(mus_zs)(theta)
    D_all = jac_mus.reshape(T + 1, Sd, n_theta)
    G_all = jac_zs.reshape(T, Sd * Ad, n_theta)
    return D_all, G_all


def _resolve_avoid_regions(env, avoid_regions):
    """(c_vec, beta, delta) -> (c_vec, beta, kappa)."""
    return [(c_vec, beta, env.kappa(delta)) for c_vec, beta, delta in avoid_regions]


def solve_socp_subproblem(env, theta_k, mus_k, Sigmas_k, D_all, G_all, tstar,
                           rho, kappa_r, alpha_r, avoid_regions, lam,
                           lam_reach=None, action_avoid_regions=(),
                           num_agents=None):
    """avoid_regions: list of (c_vec, beta, kappa) triplets over the STATE
    marginal mu_t. action_avoid_regions: list of (c_sa, beta, kappa)
    triplets over the OCCUPATION MEASURE z_t (e.g. an aggregate-action
    congestion constraint) -- uses G_all (the z_t Jacobian, already
    computed) instead of D_all, and env.action_aggregate_sigma (Corollary
    to Proposition 1) instead of env.cantelli_sigma.

    Reach is a *soft*, heavily-penalized constraint (slack xi_reach), not
    a hard one: a hard reach constraint that becomes locally infeasible at
    the current linearization point freezes the whole SOCP (reports
    infeasible every iteration regardless of trust-region size), which is
    a difference from how the sufficient condition is stated in eq:sca_reach
    but is what actually lets the optimizer keep making progress in practice.
    """
    if lam_reach is None:
        lam_reach = 2 * lam
    n_theta = D_all.shape[-1]
    dtheta = cp.Variable(n_theta)
    n_regions = len(avoid_regions)
    n_action_regions = len(action_avoid_regions)
    xi = cp.Variable((max(n_regions, 1), tstar + 1), nonneg=True)
    xi_act = cp.Variable((max(n_action_regions, 1), tstar + 1), nonneg=True)
    xi_reach = cp.Variable(nonneg=True)

    r_flat = np.array(env.R).reshape(-1)
    grad_J = sum(np.array(G_all[t]).T @ r_flat for t in range(G_all.shape[0]))

    sigma_T_tstar = float(env.cantelli_sigma(Sigmas_k[tstar], env.C_TARGET))
    cT_D_tstar = np.array(env.C_TARGET @ D_all[tstar])
    reach_rhs = alpha_r + kappa_r * sigma_T_tstar - float(env.C_TARGET @ mus_k[tstar])

    constraints = [cp.norm(dtheta, 2) <= rho,
                   cT_D_tstar @ dtheta + xi_reach >= reach_rhs]

    for i, (c_vec, beta, kappa_u) in enumerate(avoid_regions):
        for t in range(tstar + 1):
            sigma_t = float(env.cantelli_sigma(Sigmas_k[t], c_vec))
            c_D_t = np.array(c_vec @ D_all[t])
            avoid_rhs = beta - kappa_u * sigma_t - float(c_vec @ mus_k[t])
            constraints.append(c_D_t @ dtheta - xi[i, t] <= avoid_rhs)

    for i, (c_sa, beta, kappa_u) in enumerate(action_avoid_regions):
        for t in range(tstar + 1):
            sigma_t = float(env.action_aggregate_sigma(
                theta_k[t], mus_k[t], Sigmas_k[t], c_sa, num_agents))
            c_G_t = np.array(c_sa @ G_all[t]) if t < G_all.shape[0] else np.zeros(n_theta)
            z_t_val = float(c_sa @ (env.policy(theta_k[t], mus_k[t]) * mus_k[t][:, None]).reshape(-1)) \
                if t < G_all.shape[0] else 0.0
            avoid_rhs = beta - kappa_u * sigma_t - z_t_val
            constraints.append(c_G_t @ dtheta - xi_act[i, t] <= avoid_rhs)

    objective = cp.Maximize(grad_J @ dtheta - lam * cp.sum(xi)
                             - lam * cp.sum(xi_act) - lam_reach * xi_reach)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return None
    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None
    return dtheta.value


def reach_avoid_margins(env, theta, mus, Sigmas, tstar, alpha_r, kappa_r,
                         avoid_regions, action_avoid_regions=(), num_agents=None):
    """avoid_regions: (c_vec, beta, kappa) over mu_t. action_avoid_regions:
    (c_sa, beta, kappa) over z_t.
    """
    r_m = float(env.C_TARGET @ mus[tstar] - alpha_r
                - kappa_r * env.cantelli_sigma(Sigmas[tstar], env.C_TARGET))
    margins = [
        float(beta - c_vec @ mus[t] - kappa_u * env.cantelli_sigma(Sigmas[t], c_vec))
        for c_vec, beta, kappa_u in avoid_regions
        for t in range(tstar + 1)
    ]
    T = theta.shape[0]
    for c_sa, beta, kappa_u in action_avoid_regions:
        for t in range(tstar + 1):
            if t >= T:
                continue
            z_t_val = float(c_sa @ (env.policy(theta[t], mus[t]) * mus[t][:, None]).reshape(-1))
            sigma_t = float(env.action_aggregate_sigma(
                theta[t], mus[t], Sigmas[t], c_sa, num_agents))
            margins.append(beta - z_t_val - kappa_u * sigma_t)
    a_m = min(margins) if margins else float("inf")
    return r_m, a_m


def _merit(J, r_m, a_m, lam_merit):
    """L1-penalty merit function: reward, penalized by constraint violation.

    The trust-region accept/reject test must look at this (not raw reward),
    since a step that only improves J can otherwise walk straight past a
    binding safety constraint the linearized SOCP could not fully enforce
    in one step (see the caveat remark in Sec. VI.C of acc_paper.tex).
    """
    return J - lam_merit * (max(0.0, -r_m) + max(0.0, -a_m))


def _better_candidate(candidate, current):
    """Prefers a certified candidate over an uncertified one; among two
    certified candidates, prefers higher reward; among two uncertified
    ones, prefers the higher worst-case margin. Without this, "best" would
    only ever update on full certification and otherwise silently keep
    whichever t* candidate happened to be tried first, even if a later
    candidate had far better (if still uncertified) margins.
    """
    c_cert = candidate["r_m"] >= 0 and candidate["a_m"] >= 0
    b_cert = current["r_m"] >= 0 and current["a_m"] >= 0
    if c_cert != b_cert:
        return c_cert
    if c_cert:
        return candidate["J"] > current["J"]
    return min(candidate["r_m"], candidate["a_m"]) > min(current["r_m"], current["a_m"])


def sca_algorithm(env, T, num_agents, alpha_r, delta_r, avoid_regions,
                   K=20, K_tstar=3, rho0=1.0, beta_inc=1.5, beta_dec=0.5,
                   rho_max=5.0, lam=50.0, lam_reach=None, lam_merit=20.0,
                   eps=1e-4, theta0=None, verbose=False, action_avoid_regions=(),
                   tstar_candidates=None):
    """avoid_regions: list of (c_vec, beta, delta) triplets over the state
    marginal mu_t, one per independent avoid region (e.g. one per unsafe/
    congestion cell). action_avoid_regions: same triplet form but over the
    occupation measure z_t (e.g. an aggregate-action congestion constraint,
    as in the EV-charging case study) -- see evcharging.action_aggregate_sigma.

    tstar_candidates: optional explicit list of t* to try (e.g. the MF-LP's
    own t*), overriding the default auto-ranking by reach margin alone.
    That ranking only looks at the reach margin at each t, so it has no way
    to know that a *larger* t* also *extends* the avoid-constraint window --
    when the avoid window only binds up to some t* < T (as in the EV case
    study, where nothing constrains fast-charging once the deadline has
    passed), auto-ranking can pick a t* that drags in an unconstrained,
    unsafe tail. Passing the LP's own t* directly avoids this.
    """
    kappa_r = env.kappa(delta_r)
    avoid_kr = _resolve_avoid_regions(env, avoid_regions)
    action_avoid_kr = _resolve_avoid_regions(env, action_avoid_regions)
    if theta0 is None:
        theta0 = jnp.zeros((T, env.S, env.A, env.N_FEAT))

    def margins(theta, mus, Sigmas, tstar):
        return reach_avoid_margins(env, theta, mus, Sigmas, tstar, alpha_r,
                                    kappa_r, avoid_kr, action_avoid_kr, num_agents)

    if tstar_candidates is not None:
        candidates = list(tstar_candidates)
    else:
        mus0, zs0, pis0 = env.forward_sim(theta0)
        Sigmas0 = env.lyapunov_recursion(mus0, pis0, num_agents)
        margins0 = [
            float(env.C_TARGET @ mus0[t] - alpha_r
                  - kappa_r * env.cantelli_sigma(Sigmas0[t], env.C_TARGET))
            for t in range(T + 1)
        ]
        candidates = sorted(range(T + 1), key=lambda t: -margins0[t])[:K_tstar]

    best = None
    for tstar in candidates:
        theta, rho = theta0, rho0
        for it in range(K):
            mus_k, zs_k, pis_k = env.forward_sim(theta)
            Sigmas_k = env.lyapunov_recursion(mus_k, pis_k, num_agents)
            D_all, G_all = jacobians(env, theta)

            dtheta_flat = solve_socp_subproblem(
                env, theta, mus_k, Sigmas_k, D_all, G_all, tstar,
                rho, kappa_r, alpha_r, avoid_kr, lam, lam_reach,
                action_avoid_kr, num_agents)
            if dtheta_flat is None:
                rho = min(beta_inc * rho, rho_max)  # too small to be feasible: grow
                if verbose:
                    print(f"  t*={tstar} k={it:3d} infeasible, rho->{rho:.4f}")
                continue

            dtheta = jnp.array(dtheta_flat).reshape(theta.shape)
            theta_prime = theta + dtheta

            r_m_k, a_m_k = margins(theta, mus_k, Sigmas_k, tstar)
            J_k = float(env.total_reward(zs_k))
            merit_k = _merit(J_k, r_m_k, a_m_k, lam_merit)

            mus_p, zs_p, pis_p = env.forward_sim(theta_prime)
            Sigmas_p = env.lyapunov_recursion(mus_p, pis_p, num_agents)
            r_m_p, a_m_p = margins(theta_prime, mus_p, Sigmas_p, tstar)
            J_p = float(env.total_reward(zs_p))
            merit_p = _merit(J_p, r_m_p, a_m_p, lam_merit)

            if merit_p >= merit_k - eps:
                theta = theta_prime
                rho = min(beta_inc * rho, rho_max)
                accepted = True
            else:
                rho = beta_dec * rho
                accepted = False
            if verbose:
                print(f"  t*={tstar} k={it:3d} accepted={accepted} rho={rho:.4f} "
                      f"J={J_k:.3f} r_m={r_m_k:+.3f} a_m={a_m_k:+.3f}")

        mus_f, zs_f, pis_f = env.forward_sim(theta)
        Sigmas_f = env.lyapunov_recursion(mus_f, pis_f, num_agents)
        r_m, a_m = margins(theta, mus_f, Sigmas_f, tstar)
        J_f = float(env.total_reward(zs_f))
        result = dict(theta=theta, mus=mus_f, Sigmas=Sigmas_f, zs=zs_f,
                       r_m=r_m, a_m=a_m, J=J_f, tstar=tstar)

        if best is None or _better_candidate(result, best):
            best = result
        if r_m >= 0 and a_m >= 0:
            break

    return best
