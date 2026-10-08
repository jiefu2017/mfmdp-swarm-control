"""Monte Carlo evaluation of a (possibly density-feedback) time-varying
policy on the finite-N swarm, mirroring Table I/II of acc_paper.tex.

Generic over any environment module exposing the mfmdp.py interface.
"""
import numpy as np


def _sample_actions(rng, pi_sa):
    """pi_sa: (S,A) probabilities. Returns one action per state row via inverse CDF."""
    cdf = np.cumsum(pi_sa, axis=1)
    u = rng.random(pi_sa.shape[0])
    return (u[:, None] < cdf).argmax(axis=1)


def monte_carlo(env, policy_fn, T, num_agents, alpha_r, avoid_regions, tstar,
                 n_trials=2000, seed=0, start_state=0, action_avoid_regions=()):
    """policy_fn(t, mu_hat) -> pi (S,A); mu_hat is the realized empirical density.
    avoid_regions: list of (c_vec, beta, delta) triplets over the STATE
    marginal mu_t. action_avoid_regions: same triplet form but over the
    realized ACTION fractions at each step (e.g. fraction of the fleet
    choosing to fast-charge), checked for t=0..min(tstar,T-1).

    IMPORTANT: eq:avoid in the paper is a *per-timestep marginal*
    requirement (P(mu_t(U)<=beta_u) >= 1-delta_u, separately at each t),
    not a joint "never violates across the whole trajectory" event. When
    only one timestep is near the boundary those two readings nearly
    coincide, but when the constraint sits at the boundary across many
    consecutive timesteps (as in the EV congestion example) they diverge
    sharply -- P(violates at *some* t) can approach 1 even when every
    individual t is close to the 1-delta_u target. P_viol_per_region below
    is therefore the MAX over t of the per-timestep marginal violation
    rate (the quantity eq:avoid actually bounds); P_viol_hat is the looser
    "violated at any t" joint statistic, kept only as a diagnostic.

    Returns dict with J_hat, P_reach_hat, P_viol_hat (joint, diagnostic
    only), P_viol_per_region (max per-timestep marginal rate; state
    regions first, then action regions).
    """
    P_np = np.array(env.P)
    R_np = np.array(env.R)
    target_idx = np.array(env.C_TARGET) > 0
    region_idx_beta = [(np.array(c_vec) > 0, beta) for c_vec, beta, _delta in avoid_regions]
    action_region_beta = [(np.array(c_sa).reshape(env.S, env.A), beta)
                           for c_sa, beta, _delta in action_avoid_regions]

    rng = np.random.default_rng(seed)
    totals, reached, violated = [], [], []
    # per-timestep violation indicators: region_viol_by_t[i][t] = list over trials
    region_viol_by_t = [[[] for _ in range(T + 1)] for _ in region_idx_beta]
    action_viol_by_t = [[[] for _ in range(T)] for _ in action_region_beta]

    for _ in range(n_trials):
        states = np.full(num_agents, start_state, dtype=int)
        reward_sum = 0.0
        viol = False
        reach = False
        for t in range(T):
            mu_hat = np.bincount(states, minlength=env.S) / num_agents
            pi_t = np.array(policy_fn(t, mu_hat))
            actions = _sample_actions(rng, pi_t[states])
            reward_sum += R_np[states, actions].sum() / num_agents

            for i, (c_mat, beta) in enumerate(action_region_beta):
                c_a = c_mat[states, actions]  # 1 if (state,action) is in the region
                v = (t <= tstar) and (c_a.sum() / num_agents > beta)
                action_viol_by_t[i][t].append(bool(v))
                viol = viol or v

            next_states = np.empty_like(states)
            for s in np.unique(states):
                idx = np.where(states == s)[0]
                for a in np.unique(actions[idx]):
                    sub = idx[actions[idx] == a]
                    next_states[sub] = rng.choice(env.S, size=sub.size, p=P_np[s, a])
            states = next_states

            mu_next = np.bincount(states, minlength=env.S) / num_agents
            for i, (region_idx, beta) in enumerate(region_idx_beta):
                v = (t + 1 <= tstar) and (mu_next[region_idx].sum() > beta)
                region_viol_by_t[i][t + 1].append(bool(v))
                viol = viol or v
            if t + 1 == tstar and mu_next[target_idx].sum() >= alpha_r:
                reach = True

        totals.append(reward_sum)
        reached.append(reach)
        violated.append(viol)

    per_region_max = [max(float(np.mean(v)) for v in by_t if v)
                       for by_t in region_viol_by_t]
    per_action_max = [max(float(np.mean(v)) for v in by_t if v)
                       for by_t in action_viol_by_t]

    return dict(
        J_hat=float(np.mean(totals)),
        P_reach_hat=float(np.mean(reached)),
        P_viol_hat=float(np.mean(violated)),
        P_viol_per_region=per_region_max + per_action_max,
    )


def sca_policy_fn(env, theta):
    def fn(t, mu_hat):
        import jax.numpy as jnp
        return env.policy(theta[t], jnp.array(mu_hat))
    return fn


def mf_lp_policy_fn(pi_table):
    def fn(t, mu_hat):
        return pi_table[t]
    return fn
