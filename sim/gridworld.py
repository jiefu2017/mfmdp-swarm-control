"""6x6 gridworld MDP (acc_paper.tex Sec. VI.B) and its differentiable
mean-field / covariance dynamics. Same interface as mfmdp.py so
sca.py / mf_lp.py / evaluate.py run unchanged on either environment.

State s = x*GRID + y (row-major over (x,y) in [0,GRID)^2).
Actions: 0=up (y+1), 1=down (y-1), 2=left (x-1), 3=right (x+1), 4=stay.
Moves off the grid are clipped (agent stays in place instead).
Transition noise: intended action w.p. 0.9; w.p. 0.1 a uniformly
random *other* action is executed instead (not specified exactly in
the paper text -- this is the natural reading of "slip 0.1").
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

GRID = 6
S, A = GRID * GRID, 5
N_FEAT = 5  # phi = [1, dist_to_reach_target, dist_to_nearest_unsafe, mu(target), mu(unsafe)]

START = (5, 0)
REACH_TARGETS = [(0, 3), (0, 0), (0, 4)]  # reach set, reward 0.5 each, past the wall
REWARD_ONLY_TARGET = (0, 5)   # s1: reward 1.0, NOT in reach set, also past the wall
# Wall at x=2 with three physically-impassable cells and three openings.
# WALL_CELLS block movement entirely (true obstacles); UNSAFE_CELLS are the
# openings themselves -- ordinary passable states, but every route must pass
# through one of them, and the avoid constraint tracks their *instantaneous*
# occupancy (congestion), not cumulative absorption: unlike an absorbing
# sink, mu_t(unsafe) here can rise and fall as the crossing wave passes
# through, rather than growing monotonically over the whole horizon.
WALL_CELLS = [(2, 0), (2, 2), (2, 4)]
UNSAFE_CELLS = [(2, 1), (2, 3), (2, 5)]

P_INTENDED = 0.9
_WALL_XY = set(WALL_CELLS)


def _idx(x, y):
    return x * GRID + y


def _xy(s):
    return s // GRID, s % GRID


def _move(x, y, a):
    if a == 0:
        nx, ny = x, min(y + 1, GRID - 1)
    elif a == 1:
        nx, ny = x, max(y - 1, 0)
    elif a == 2:
        nx, ny = max(x - 1, 0), y
    elif a == 3:
        nx, ny = min(x + 1, GRID - 1), y
    else:  # a == 4: stay
        nx, ny = x, y
    if (nx, ny) in _WALL_XY:
        return x, y  # wall blocks movement; agent stays put instead
    return nx, ny


def _build_transition_tensor():
    P = np.zeros((S, A, S))
    for s in range(S):
        x, y = _xy(s)
        for a in range(A):
            for a_eff in range(A):
                prob = P_INTENDED if a_eff == a else (1.0 - P_INTENDED) / (A - 1)
                x2, y2 = _move(x, y, a_eff)
                P[s, a, _idx(x2, y2)] += prob
    return P


P = jnp.array(_build_transition_tensor())

R = np.zeros((S, A))
R[_idx(*REWARD_ONLY_TARGET), :] = 1.0
for _cell in REACH_TARGETS:
    R[_idx(*_cell), :] = 0.5
R = jnp.array(R)

MU0 = jnp.zeros(S).at[_idx(*START)].set(1.0)

C_TARGET = jnp.zeros(S)
for _cell in REACH_TARGETS:
    C_TARGET = C_TARGET.at[_idx(*_cell)].set(1.0)
C_UNSAFE = jnp.zeros(S)
for _cell in UNSAFE_CELLS:
    C_UNSAFE = C_UNSAFE.at[_idx(*_cell)].set(1.0)

# Per-cell indicator vectors, for per-cell (rather than aggregate) congestion
# constraints: one Cantelli constraint per unsafe/gap cell instead of one
# constraint on their sum.
C_UNSAFE_CELLS = [jnp.zeros(S).at[_idx(*_cell)].set(1.0) for _cell in UNSAFE_CELLS]

_XY = jnp.array([[s // GRID, s % GRID] for s in range(S)], dtype=jnp.float64)
_REACH_XY = jnp.array(REACH_TARGETS, dtype=jnp.float64)
_UNSAFE_XY = jnp.array(UNSAFE_CELLS, dtype=jnp.float64)
_MAX_DIST = 2 * (GRID - 1)


def _dist_to_reach(s):
    dists = jnp.sum(jnp.abs(_XY[s][None, :] - _REACH_XY), axis=1)
    return jnp.min(dists) / _MAX_DIST


def _dist_to_unsafe(s):
    dists = jnp.sum(jnp.abs(_XY[s][None, :] - _UNSAFE_XY), axis=1)
    return jnp.min(dists) / _MAX_DIST


_DIST_REACH = jnp.array([_dist_to_reach(s) for s in range(S)])
_DIST_UNSAFE = jnp.array([_dist_to_unsafe(s) for s in range(S)])


def feature_map(s, mu):
    return jnp.array([1.0, _DIST_REACH[s], _DIST_UNSAFE[s], mu @ C_TARGET, mu @ C_UNSAFE])


def _all_features(mu):
    """Vectorized feature_map over all states: returns (S, N_FEAT)."""
    mu_t = mu @ C_TARGET
    mu_u = mu @ C_UNSAFE
    ones = jnp.ones(S)
    return jnp.stack([ones, _DIST_REACH, _DIST_UNSAFE,
                       jnp.full(S, mu_t), jnp.full(S, mu_u)], axis=1)


def policy(theta_t, mu):
    """theta_t: (S, A, N_FEAT). Returns pi(a|s,mu): (S, A)."""
    phi = _all_features(mu)  # (S, N_FEAT)
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


def _bfs_distance(goal_xy_list):
    """Multi-source BFS: shortest-path distance to the *nearest* of the given
    goal cells, over non-wall cells (walls are the only physically-blocked
    cells now; unsafe/gap cells are ordinary passable states and are not
    avoided for path-planning purposes -- SCA, not the warm start, is
    responsible for managing congestion through them).
    """
    from collections import deque
    dist = np.full(S, np.inf)
    q = deque()
    for goal_xy in goal_xy_list:
        goal = _idx(*goal_xy)
        dist[goal] = 0
        q.append(goal)
    while q:
        s = q.popleft()
        x, y = _xy(s)
        for a in range(4):
            x2, y2 = _move(x, y, a)
            s2 = _idx(x2, y2)
            if s2 == s:  # blocked by a wall (or already at goal's own cell)
                continue
            if dist[s2] == np.inf:
                dist[s2] = dist[s] + 1
                q.append(s2)
    return dist


def warm_start_theta(T, bias=3.0):
    """Geometry-aware warm start: bias each state's greedy action toward the
    reach target via BFS shortest-path distance (routing around walls),
    using the constant bias feature (index 0). Needed because a uniform
    (theta=0) policy never meaningfully approaches the target, so the t*
    candidate ranking has nothing informative to rank.
    """
    theta0 = np.zeros((T, S, A, N_FEAT))
    dist = _bfs_distance(REACH_TARGETS)
    for s in range(S):
        x, y = _xy(s)
        best_a, best_d = 4, dist[s]
        for a in range(4):
            x2, y2 = _move(x, y, a)
            s2 = _idx(x2, y2)
            if dist[s2] < best_d:
                best_a, best_d = a, dist[s2]
        theta0[:, s, best_a, 0] = bias
    return jnp.array(theta0)
