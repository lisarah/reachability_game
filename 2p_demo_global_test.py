import numpy as np
from itertools import product
import time


# ============================================================
# 1. GRID TRANSITION MODEL
# ============================================================

def grid_transitions(M, N, p=0.8, with_stay=True):
    """
    Build transition tensor P[s', s, a]
    """
    A = 5 if with_stay else 4
    S = M * N
    P = np.zeros((S, S, A))

    def neighbors(i, j):
        nbrs = {}
        nbrs[0] = (i, j-1) if j > 0 else (i, j)
        nbrs[1] = (i, j+1) if j < N-1 else (i, j)
        nbrs[2] = (i-1, j) if i > 0 else (i, j)
        nbrs[3] = (i+1, j) if i < M-1 else (i, j)
        if with_stay:
            nbrs[4] = (i, j)
        return nbrs

    for i in range(M):
        for j in range(N):
            s = i * N + j
            nbrs = neighbors(i, j)

            for a in nbrs:
                ni, nj = nbrs[a]
                s_main = ni * N + nj

                # main direction probability
                P[s_main, s, a] += p

                # distribute noise across valid neighbors
                valid_states = list(
                    ni * N + nj for ni, nj in nbrs.values()
                )
                noise = (1 - p) / len(valid_states)

                for sp in valid_states:
                    P[sp, s, a] += noise

    return P


# ============================================================
# 2. MAKE STATE ABSORBING
# ============================================================

def make_absorbing(P, state):
    S, _, A = P.shape
    for a in range(A):
        P[:, state, a] = 0
        P[state, state, a] = 1.0
    return P


# ============================================================
# 3. BUILD JOINT MDP
# ============================================================

def build_joint_mdp(P1, P2):
    S_single, _, A_single = P1.shape

    joint_states = list(product(range(S_single), repeat=2))
    joint_actions = list(product(range(A_single), repeat=2))


    state_to_idx = {s: i for i, s in enumerate(joint_states)}
    idx_to_state = {i: s for s, i in state_to_idx.items()}

    S_joint = len(joint_states)
    A_joint = len(joint_actions)

    P_joint = np.zeros((S_joint, S_joint, A_joint))

    for s_idx, (s1, s2) in enumerate(joint_states):
        for a_idx, (a1, a2) in enumerate(joint_actions):

            for s1_next in range(S_single):
                for s2_next in range(S_single):

                    prob = (
                        P1[s1_next, s1, a1] *
                        P2[s2_next, s2, a2]
                    )

                    if prob > 0:
                        sp_idx = state_to_idx[(s1_next, s2_next)]
                        P_joint[sp_idx, s_idx, a_idx] += prob

    return P_joint, state_to_idx, idx_to_state, joint_actions


# ============================================================
# 4. GLOBAL FINITE-HORIZON REACH-AVOID VALUE ITERATION
# ============================================================

def global_reachability(P_joint, T, goal_states, collision_states):
    S_joint, _, A_joint = P_joint.shape

    V = np.zeros((S_joint, T+1))
    policy = np.zeros((S_joint, T), dtype=int)

    # Terminal condition
    for s in goal_states:
        V[s, T] = 1.0

    for t in reversed(range(T)):
        print(f"Computing time step {t}")

        for s in range(S_joint):

            # Collision state → absorbing failure
            if s in collision_states:
                V[s, t] = 0
                continue

            Q = np.zeros(A_joint)

            for a in range(A_joint):
                Q[a] = P_joint[:, s, a].dot(V[:, t+1])

            best_a = np.argmax(Q)
            V[s, t] = Q[best_a]
            policy[s, t] = best_a

    return V, policy


# ============================================================
# 5. SIMULATION
# ============================================================

def simulate(policy, P1, P2, joint_actions,
             state_to_idx, idx_to_state,
             init_state, T):

    current = state_to_idx[init_state]
    traj = [init_state]

    S_single = P1.shape[0]

    for t in range(T):
        a_idx = policy[current, t]
        a1, a2 = joint_actions[a_idx]

        s1, s2 = idx_to_state[current]

        s1_next = np.random.choice(
            range(S_single),
            p=P1[:, s1, a1]
        )

        s2_next = np.random.choice(
            range(S_single),
            p=P2[:, s2, a2]
        )

        next_state = (s1_next, s2_next)
        traj.append(next_state)
        current = state_to_idx[next_state]

    return traj


# ============================================================
# 6. MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    Rows = 9
    Columns = 5
    T = 8

    P1 = grid_transitions(Rows, Columns, p=0.8)
    P2 = grid_transitions(Rows, Columns, p=0.8)


    S_single = Rows * Columns

    rng = np.random.default_rng()
    states = rng.choice(range(S_single), size=4, replace=False)

    targ1, targ2 = states[0], states[1]
    init1, init2 = states[2], states[3]

    print("Target states:", targ1, targ2)
    print("Initial states:", init1, init2)

    # Make targets absorbing BEFORE building joint
    P1 = make_absorbing(P1, targ1)
    P2 = make_absorbing(P2, targ2)

    # Build joint MDP
    P_joint, state_to_idx, idx_to_state, joint_actions = \
        build_joint_mdp(P1, P2)

    S_joint = len(state_to_idx)

    goal_joint = [state_to_idx[(targ1, targ2)]]

    # Marks all joint states where s1=s2 as a collision state
    collision_states = [
        state_to_idx[(s, s)]
        for s in range(S_single)
    ]

    # Make joint collision states absorbing
    for s in collision_states:
        for a in range(len(joint_actions)):
            P_joint[:, s, a] = 0
            P_joint[s, s, a] = 1.0

    start = time.time()
    V, policy = global_reachability(
        P_joint,
        T,
        goal_joint,
        collision_states
    )
    print("Solve time:", time.time() - start)

    # Simulate
    trajectory = simulate(
        policy,
        P1,
        P2,
        joint_actions,
        state_to_idx,
        idx_to_state,
        (init1, init2),
        T
    )

    print("\nTrajectory:")
    for t, s in enumerate(trajectory):
        print(f"t={t}: {s}")

    print("\nInitial success probability:",
          V[state_to_idx[(init1, init2)], 0])