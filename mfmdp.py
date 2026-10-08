"""4-state stochastic graph MDP (Fig. 1 of acc_paper.tex) and its
differentiable mean-field / covariance dynamics.

States: 0=s0, 1=s1, 2=s2 (target), 3=s3 (unsafe). Both s2, s3 absorbing.
Actions: 0=a0 (risky, reward 3), 1=a1 (cautious, reward 1).
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

S, A = 4, 2
TARGET, UNSAFE = 2, 3
N_FEAT = 3  # phi(mu) = [1, mu(target), mu(unsafe)]

# P[s, a, s'] = P(s' | s, a)
P = np.zeros((S, A, S))
P[0, 0] = [0.0, 0.65, 0.0, 0.35]   # s0, risky
P[0, 1] = [0.0, 0.95, 0.0, 0.05]   # s0, cautious
P[1, 0] = [0.0, 0.0, 0.65, 0.35]   # s1, risky
P[1, 1] = [0.0, 0.0, 0.95, 0.05]   # s1, cautious
P[2, :] = [0.0, 0.0, 1.0, 0.0]     # s2 absorbing
P[3, :] = [0.0, 0.0, 0.0, 1.0]     # s3 absorbing
P = jnp.array(P)

# R[s, a]
R = np.zeros((S, A))
R[0, 0] = 3.0
R[0, 1] = 1.0
R[1, 0] = 3.0
R[1, 1] = 1.0
R = jnp.array(R)

MU0 = jnp.array([1.0, 0.0, 0.0, 0.0])

C_TARGET = jnp.array([0.0, 0.0, 1.0, 0.0])
C_UNSAFE = jnp.array([0.0, 0.0, 0.0, 1.0])


def feature_map(mu):
    """phi(mu) in R^{N_FEAT}: bias, target density, unsafe density."""
    return jnp.array([1.0, mu[TARGET], mu[UNSAFE]])


def policy(theta_t, mu):
    """theta_t: (S, A, N_FEAT). Returns pi(a|s,mu): (S, A)."""
    phi = feature_map(mu)
    logits = jnp.einsum("saf,f->sa", theta_t, phi)
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
    """theta: (T, S, A, N_FEAT). Returns mus (T+1,S), zs (T,S,A), pis (T,S,A)."""
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
    """A_t[s', s] = sum_a pi_t(a|s) P(s'|s,a)."""
    return jnp.einsum("sa,sap->ps", pi_t, P)


def multinomial_noise_cov(A_t, mu_t):
    Lambda = jnp.zeros((S, S))
    for s in range(S):
        col = A_t[:, s]
        Lambda = Lambda + mu_t[s] * (jnp.diag(col) - jnp.outer(col, col))
    return Lambda


def lyapunov_recursion(mus, pis, num_agents):
    """Propagates Sigma_t (Prop. 1). Returns Sigmas: (T+1, S, S), Sigma_0=0."""
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
