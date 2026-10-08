"""Overnight EV-charging aggregator MDP: a power-systems case study for
acc_paper.tex, illustrating that the framework extends beyond spatial
"unsafe regions" to a congestion constraint on the *occupation measure*
(aggregate fast-charge action), not the state marginal.

State s in {0,...,10}: state of charge (SOC) in 10% bins.
Actions: 0=fast (reward 3, +2 bins w.p. 0.6 else +1), 1=slow (reward 1,
+1 bin w.p. 0.9 else unchanged). State 10 (full) is absorbing.

Reach set: SOC >= 80% (states 8,9,10) by departure deadline t*.
Avoid/congestion: the aggregate fraction of the fleet fast-charging
*simultaneously* must stay under beta_u (a transformer capacity budget)
w.p. >= 1-delta_u -- a constraint on z_t(.,fast), not on mu_t.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

S, A = 11, 3  # SOC bins 0..10 (0%,...,100%); 0=fast, 1=slow, 2=idle
N_FEAT = 3    # phi = [1, s/10, mu(target)]
FAST, SLOW, IDLE = 0, 1, 2
FULL = S - 1
TARGET_SOC = 8  # >= 80%

P_FAST_2BIN = 0.6
P_SLOW_1BIN = 0.9


def _build_transition_tensor():
    P = np.zeros((S, A, S))
    for s in range(S):
        if s == FULL:
            P[s, :, FULL] = 1.0
            continue
        P[s, FAST, min(s + 2, FULL)] += P_FAST_2BIN
        P[s, FAST, min(s + 1, FULL)] += 1 - P_FAST_2BIN
        P[s, SLOW, min(s + 1, FULL)] += P_SLOW_1BIN
        P[s, SLOW, s] += 1 - P_SLOW_1BIN
        P[s, IDLE, s] = 1.0  # no charging: SOC unchanged, deterministic
    return P


P = jnp.array(_build_transition_tensor())

R = np.zeros((S, A))
R[:FULL, FAST] = 3.0
R[:FULL, SLOW] = 1.0
# IDLE (no charging): reward 0, left at the zeros default.
R = jnp.array(R)

MU0 = jnp.zeros(S).at[0].set(1.0)

C_TARGET = jnp.zeros(S)
for _s in range(TARGET_SOC, S):
    C_TARGET = C_TARGET.at[_s].set(1.0)

# Aggregate-action indicator over the flattened (s,a) occupation measure,
# used for the congestion (avoid) constraint instead of a state indicator.
C_FAST_SA = jnp.zeros(S * A)
for _s in range(S):
    C_FAST_SA = C_FAST_SA.at[_s * A + FAST].set(1.0)

_SOC_NORM = jnp.arange(S, dtype=jnp.float64) / (S - 1)


def _all_features(mu):
    mu_t = mu @ C_TARGET
    ones = jnp.ones(S)
    return jnp.stack([ones, _SOC_NORM, jnp.full(S, mu_t)], axis=1)


def policy(theta_t, mu):
    """theta_t: (S, A, N_FEAT). Returns pi(a|s,mu): (S, A)."""
    phi = _all_features(mu)
    logits = jnp.einsum("saf,sf->sa", theta_t, phi)
    return jax.nn.softmax(logits, axis=-1)


def occupation_measure(theta_t, mu):
    pi = policy(theta_t, mu)
    z = pi * mu[:, None]
    return z, pi


def mf_step(theta_t, mu):
    z, pi = occupation_measure(theta_t, mu)
    mu_next = jnp.einsum("sa,sap->p", z, P)
    return mu_next, z, pi


def forward_sim(theta, mu0=MU0):
    T = theta.shape[0]
    mu = mu0
    mus = [mu]
    zs, pis = [], []
    for t in range(T):
        mu, z, pi = mf_step(theta[t], mu)
        mus.append(mu)
        zs.append(z)
        pis.append(pi)
    return jnp.stack(mus), jnp.stack(zs), jnp.stack(pis)


def effective_transition_matrix(pi_t):
    return jnp.einsum("sa,sap->ps", pi_t, P)


def multinomial_noise_cov(A_t, mu_t):
    Lambda = jnp.zeros((S, S))
    for s in range(S):
        col = A_t[:, s]
        Lambda = Lambda + mu_t[s] * (jnp.diag(col) - jnp.outer(col, col))
    return Lambda


def lyapunov_recursion(mus, pis, num_agents):
    T = pis.shape[0]
    Sigma = jnp.zeros((S, S))
    Sigmas = [Sigma]
    for t in range(T):
        A_t = effective_transition_matrix(pis[t])
        Lambda_t = multinomial_noise_cov(A_t, mus[t])
        Sigma = A_t @ Sigma @ A_t.T + Lambda_t / num_agents
        Sigmas.append(Sigma)
    return jnp.stack(Sigmas)


def cantelli_sigma(Sigma_t, c_region):
    val = c_region @ Sigma_t @ c_region
    return jnp.sqrt(jnp.maximum(val, 0.0))


def kappa(delta):
    return float(np.sqrt((1.0 - delta) / delta))


def total_reward(zs):
    return jnp.sum(zs * R[None, :, :])


def action_aggregate_sigma(theta_t, mu_t, Sigma_t, c_sa, num_agents):
    """Cantelli std. dev. of an aggregate ACTION functional c_sa^T z_t^N
    (e.g. fraction of the fleet fast-charging), via the delta method:
    a direct Bernoulli/multinomial term (the z_t-analogue of Prop. 1's
    Lambda_t, evaluated once per state -- no new recursion needed) plus
    the *propagated* term through the already-computed state-density
    covariance Sigma_t = Cov[mu_t^N], using the density-feedback Jacobian
    d(c_sa^T z_t)/d(mu_t) obtained directly via autodiff.
    Corollary to Proposition 1, extending it from the state marginal
    mu_t^N to a linear functional of the occupation measure z_t^N.
    """
    c_mat = c_sa.reshape(S, A)
    pi_t = policy(theta_t, mu_t)
    p_c = jnp.sum(pi_t * c_mat, axis=1)  # P(c=1 | s), shape (S,)
    direct_var = jnp.sum(mu_t * p_c * (1 - p_c)) / num_agents

    def g(mu):
        pi = policy(theta_t, mu)
        z = pi * mu[:, None]
        return jnp.sum(z.reshape(-1) * c_sa)

    grad_g = jax.grad(g)(mu_t)
    propagated_var = grad_g @ Sigma_t @ grad_g
    return jnp.sqrt(jnp.maximum(direct_var + propagated_var, 0.0))


def warm_start_theta(T, bias=3.0):
    """Simple geometry-aware warm start: bias each state toward whichever
    action (fast/slow) makes the most SOC progress, ignoring congestion
    (SCA is responsible for learning to stagger fast-charging).
    """
    theta0 = np.zeros((T, S, A, N_FEAT))
    for s in range(S - 1):
        theta0[:, s, FAST, 0] = bias
    return jnp.array(theta0)
