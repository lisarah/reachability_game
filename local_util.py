# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 08:11:52 2026

@author: adamc
"""


import numpy as np
import MDP_algorithms.mdp_state_action as mdp
import MDP_algorithms.value_iteration as vi
import util as ut
import time
from itertools import permutations
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation


class Local_MDP:

    def __init__(self, Rows, Columns, T, player_num):
        self.Rows = Rows
        self.Columns = Columns
        self.T = T
        self.player_num = player_num

    # ------------------------------------------------------------------
    # Problem setup
    # ------------------------------------------------------------------

    def create_MDP(self, action_entropy, targs_x0s):
        """
        Build individual transition models and extract targets / initial states.
        Mirrors create_joint_MDP in Global_MDP but without forming the joint space.
        """
        print("Starting local MDP construction...")
        time_start = time.time()

        # Individual transition tensors P[s', s, a] for each player
        self.Ps = [
            mdp.transitions(self.Rows, self.Columns, p=action_entropy, with_stay=True)
            for _ in range(self.player_num)
        ]

        S, _, A = self.Ps[0].shape
        self.S = S
        self.A = A

        # targs_x0s layout: first player_num entries = targets,
        #                   next  player_num entries = initial states
        self.targ_raw_inds = targs_x0s[:self.player_num]
        self.x_0s          = targs_x0s[self.player_num:]

        print("-" * 40)
        print(f"Target nodes:  {self.targ_raw_inds}")
        print(f"Starting nodes:{self.x_0s}")

        # Make target states absorbing sinks (action index 4 = stay)
        for p in range(self.player_num):
            self.Ps[p][:, self.targ_raw_inds[p], 4] = 0.0
            self.Ps[p][self.targ_raw_inds[p], self.targ_raw_inds[p], 4] = 1.0

        # Reachable set: dict (s, a) -> list of reachable s'
        self.reachable_set = mdp.reachable_set(self.Ps[0])

        # All states reachable from s under any action — used in best response
        self.all_reachable = {s: set() for s in range(S)}
        for s in range(S):
            for a in range(A):
                self.all_reachable[s].update(self.reachable_set[(s, a)])

        time_end = time.time()
        print(f"Time to construct local MDP: {np.round(time_end - time_start, 2)} seconds.")

    # ------------------------------------------------------------------
    # Algorithm 2: Local Feedback Best Response for player i
    # ------------------------------------------------------------------

    def _best_response(self, i, pols, rhos):
        """
        Computes player i's best response policy given fixed opponent policies.

        Implements Algorithm 2 from the paper:
          - Forward pass: compute opponent occupancy measures rho^t_{-i}
          - Backward pass: compute W^t_i(s_i) — player i's marginalized
            value function — via the multiplicative DP over the joint space,
            averaging out opponents with their occupancy measures.

        Parameters
        ----------
        i     : int         — index of the player computing best response
        pols  : list of (S, T) arrays — current policies for all players
        rhos  : list of (S, T+1) arrays — current occupancy measures

        Returns
        -------
        W_i   : (S, T+1) array — player i's local value function
        pi_i  : (S, T)   array — player i's best response policy
        """
        N     = self.player_num
        S     = self.S
        T     = self.T
        N_opp = [j for j in range(N) if j != i]

        # ------ Forward pass: occupancy measures for opponents (lines 4-9) ------
        # rho^{t+1}_j(s'_j) = sum_{s_j} P_j(s'_j | s_j, pi^t_j(s_j)) * rho^t_j(s_j)
        # We propagate each opponent's occupancy forward using their current policy.
        # rhos[j] is already computed externally and passed in — we use it directly.

        # ------ Backward pass: multiplicative DP (lines 11-22) ------
        # V^T_pi(s) = prod_j X_j(s^T_j) * prod_{i,j} Y_ij(s_i, s_j)
        # W^t_i(s_i) = sum_{s_{-i}} rho^t_{-i}(s_{-i}) * V^t_pi(s_i, s_{-i})

        # Terminal joint value function: sparse dict over collision-free,
        # all-at-target joint states. Value = 1, all others = 0.
        # This is V^T_pi from equation (20) in the paper.
        v_ind  = 0
        v_prev = 1
        V = [{}, {}]

        S_list = list(range(S))

        # Initialize V^T: all agents at their respective targets, no collision
        for s in permutations(S_list, N):
            at_targets   = all(s[p] == self.targ_raw_inds[p] for p in range(N))
            # permutations already guarantees no repeated states (Y=1 satisfied)
            if at_targets:
                V[v_ind][s] = 1.0

        # Build tau[t][j]: joint (s'_j, s_j) -> probability mass
        # = P_j(s'_j | s_j, pi_j(s_j)) * rho_j(s_j, t)
        # This encodes the weighted two-step occupancy for each opponent.
        tau = [{j: {} for j in N_opp} for _ in range(T)]
        for t in range(T - 1, -1, -1):
            for j in N_opp:
                for sj in S_list:
                    if rhos[j][sj, t] > 1e-10:
                        aj = int(pols[j][sj, t])
                        for hat_sj in self.reachable_set[(sj, aj)]:
                            key = (hat_sj, sj)
                            tau[t][j][key] = (
                                self.Ps[j][hat_sj, sj, aj] * rhos[j][sj, t]
                            )

        # W_i(s_i, t): player i's marginalized value function over individual states
        # Shape: (S, T+1)
        W_i  = np.zeros((S, T + 1))
        pi_i = np.zeros((S, T), dtype=int)

        # Terminal W^T_i(s_i) = sum_{s_{-i}} rho^T_{-i}(s_{-i}) * V^T(s_i, s_{-i})
        # (line 12 of Algorithm 2)
        for s in V[v_ind].keys():
            # s is a full joint state tuple; s[i] is player i's component
            opp_rho_prod = np.prod([rhos[j][s[j], T] for j in N_opp])
            W_i[s[i], T] += opp_rho_prod * V[v_ind][s]

        # Backward induction: t = T-1 down to 0
        for t in range(T - 1, -1, -1):
            eps   = (1e-2) ** (0.75 * t + 3)
            v_ind  = (v_ind  + 1) % 2
            v_prev = (v_ind  + 1) % 2

            # Build new V^t from V^{t+1} (lines 17-21 of Algorithm 2)
            opp_transition_list = [tau[t][j].keys() for j in N_opp]

            for si in S_list:
                # eV[hat_si] accumulates the expected future value for player i
                # transitioning to hat_si, averaged over opponents' joint transitions
                eV = {hat_si: 0.0 for hat_si in self.all_reachable[si]}

                # Insert player i's possible transitions at position i in the list
                opp_transition_list.insert(
                    i, [(hat_si, si) for hat_si in self.all_reachable[si]]
                )

                # Iterate over all joint next-state combinations (s', s) pairs
                for hat_s_s in ut.cartesian_product(opp_transition_list):
                    hat_s = tuple(hss[0] for hss in hat_s_s)

                    # Y indicator: no collision in next joint state
                    no_collision = len(set(hat_s)) == N

                    if no_collision and hat_s in V[v_prev]:
                        # Opponent joint density: prod_{j != i} tau[t][j][(hat_sj, sj)]
                        opp_density = np.prod([
                            tau[t][j][hat_s_s[j if j < i else j][0],
                                      hat_s_s[j if j < i else j][1]]
                            if False else  # use dict key directly
                            tau[t][j][hat_s_s[j if j < i else j]]
                            for j in N_opp
                        ])
                        # Simpler direct indexing:
                        opp_density = 1.0
                        for j in N_opp:
                            idx = j if j < i else j  # position in hat_s_s after insert
                            opp_density *= tau[t][j][hat_s_s[idx]]

                        if opp_density >= 1e-12:
                            eV[hat_s_s[i][0]] += opp_density * V[v_prev][hat_s]

                opp_transition_list.pop(i)

                # Q[a] = sum_{s'_i} P_i(s'_i | s_i, a) * eV[s'_i]
                # (the argmax over Q gives pi*_i(s_i, t) — equation 24)
                Q = [
                    np.sum([
                        self.Ps[i][hat_si, si, a] * eV[hat_si]
                        for hat_si in self.reachable_set[(si, a)]
                    ])
                    for a in range(self.A)
                ]
                pi_i[si, t] = int(np.argmax(Q))

            # Update V^t for all collision-free joint states s
            # V^t(s) = Y(s) * sum_{s'} prod_j P_j(s'_j|s_j,pi_j) * V^{t+1}(s')
            for s in permutations(S_list, N):
                prod_rho = np.prod([rhos[j][s[j], t] for j in N_opp])
                if prod_rho <= eps:
                    if s in V[v_ind]:
                        del V[v_ind][s]
                    continue

                hats_list     = [self.reachable_set[(s[p], int(pols[p][s[p], t]))]
                                 for p in range(N)]
                reachable_hats = ut.cartesian_product(hats_list)
                valid_hats     = V[v_prev].keys() & reachable_hats

                V_s = sum(
                    V[v_prev][hat_s] * np.prod([
                        self.Ps[p][hat_s[p], s[p], int(pols[p][s[p], t])]
                        for p in range(N)
                    ])
                    for hat_s in valid_hats
                )

                if V_s > eps:
                    V[v_ind][s] = V_s
                elif s in V[v_ind]:
                    del V[v_ind][s]

            # W^t_i(s_i) = max_{a_i} sum_{s'_i} P_i(s'_i|s_i,a_i) * W^{t+1}_i(s'_i)
            # Already stored implicitly via pi_i above.
            # Store W^t_i for metric computation.
            for si in S_list:
                a_star = pi_i[si, t]
                W_i[si, t] = np.sum([
                    self.Ps[i][hat_si, si, a_star] * W_i[hat_si, t + 1]
                    for hat_si in self.reachable_set[(si, a_star)]
                ])

        V[0].clear()
        V[1].clear()
        return W_i, pi_i

    # ------------------------------------------------------------------
    # Algorithm 3: Iterative Best Response -> Nash equilibrium
    # ------------------------------------------------------------------

    def local_value_iteration(self, BR_iter=10, tol=1e-5):
        """
        Implements Algorithm 3: Iterative Best Response.

        Each agent computes its best response in turn (round-robin) until
        the potential value converges, yielding a deterministic Nash equilibrium
        in the local feedback policy space.

        Parameters
        ----------
        BR_iter : int   — maximum number of full rounds (each player updates once)
        tol     : float — convergence threshold on potential value change
        """
        print("-" * 40)
        print("Starting iterative best response (Algorithm 3)...")
        time_start = time.time()

        S = self.S
        T = self.T
        N = self.player_num

        # Initialize policies: flat (stay) policy — action 4
        self.pols = [np.full((S, T), 4, dtype=int) for _ in range(N)]

        # Initialize occupancy measures from flat policy
        self.rhos = [
            mdp.occupancy(self.pols[p], self.Ps[p], self.x_0s[p])
            for p in range(N)
        ]

        # Store W functions for each player (used in metric computation)
        self.Ws = [np.zeros((S, T + 1)) for _ in range(N)]

        prev_potential = -np.inf

        for k in range(BR_iter * N):
            i = k % N
            print(f"\nBR iteration {k // N + 1}, updating player {i}...")

            W_i, pi_i = self._best_response(i, self.pols, self.rhos)

            self.Ws[i]   = W_i
            self.pols[i] = pi_i
            self.rhos[i] = mdp.occupancy(self.pols[i], self.Ps[i], self.x_0s[i])

            # Check convergence every full round (all N players updated)
            if (k + 1) % N == 0:
                current_potential = self._compute_potential_from_W()
                delta = abs(current_potential - prev_potential)
                print(f"  Potential = {np.round(current_potential, 8)}, delta = {np.round(delta, 8)}")
                if delta < tol:
                    print(f"  Converged after {k // N + 1} full rounds.")
                    break
                prev_potential = current_potential

        time_end = time.time()
        self.time_elapsed = np.round(time_end - time_start, 2)
        print(f"\nTime for local value iteration: {self.time_elapsed} seconds.")

    # ------------------------------------------------------------------
    # Internal potential computation (used for convergence check)
    # ------------------------------------------------------------------

    def _compute_potential_from_W(self):
        """
        F(pi) = sum_s prod_i rho^0_i(s_i) * V^0_pi(s)
        Approximated from W_i values at t=0 weighted by initial occupancy.
        Uses the joint value function structure from Proposition 1.
        """
        S = self.S
        N = self.player_num

        # Build sparse V^0 from current policies
        V = {}
        S_list = list(range(S))

        # Terminal V^T: targets, no collision
        for s in permutations(S_list, N):
            if all(s[p] == self.targ_raw_inds[p] for p in range(N)):
                V[s] = 1.0

        for t in range(self.T - 1, -1, -1):
            V_new = {}
            for s in permutations(S_list, N):
                hats_list      = [self.reachable_set[(s[p], int(self.pols[p][s[p], t]))]
                                  for p in range(N)]
                reachable_hats = ut.cartesian_product(hats_list)
                valid_hats     = V.keys() & reachable_hats
                V_s = sum(
                    V[hat_s] * np.prod([
                        self.Ps[p][hat_s[p], s[p], int(self.pols[p][s[p], t])]
                        for p in range(N)
                    ])
                    for hat_s in valid_hats
                )
                if V_s > 1e-10:
                    V_new[s] = V_s
            V = V_new

        total_V = sum(
            V[s] * np.prod([self.rhos[p][s[p], 0] for p in range(N)])
            for s in V.keys()
        )
        return total_V

    # ------------------------------------------------------------------
    # Metric computation (paper definitions)
    # ------------------------------------------------------------------

    def get_potential(self):
        """
        F(pi*_1,...,pi*_N) = E[R(tau_1,...,tau_N) | tau_i ~ h_i(pi*_i)]

        The expected joint reach-avoid objective (Equation 7 of the paper).
        Computed as the probability-weighted sum of V^0 over initial states.
        Since initial states are deterministic, this is just V^0(x_0).
        """
        S      = self.S
        N      = self.player_num
        S_list = list(range(S))

        # Rebuild V^0 under converged policies using multiplicative DP
        V = {}
        for s in permutations(S_list, N):
            if all(s[p] == self.targ_raw_inds[p] for p in range(N)):
                V[s] = 1.0

        for t in range(self.T - 1, -1, -1):
            V_new = {}
            for s in permutations(S_list, N):
                hats_list      = [self.reachable_set[(s[p], int(self.pols[p][s[p], t]))]
                                  for p in range(N)]
                reachable_hats = ut.cartesian_product(hats_list)
                valid_hats     = V.keys() & reachable_hats
                V_s = sum(
                    V[hat_s] * np.prod([
                        self.Ps[p][hat_s[p], s[p], int(self.pols[p][s[p], t])]
                        for p in range(N)
                    ])
                    for hat_s in valid_hats
                )
                if V_s > 1e-10:
                    V_new[s] = V_s
            V = V_new

        start_s = tuple(self.x_0s)
        self.potential_value = V.get(start_s, 0.0)

    def get_collision_likelihood(self):

        #Equation (27) of the paper: probability that at least one collision occurs at any timestep along the trajectory.


        S      = self.S
        N      = self.player_num
        S_list = list(range(S))

        # mu_safe[(s, t)] = probability of being in collision-free joint state s at t
        mu_safe = {}
        start_s = tuple(self.x_0s)

        # Check if agents start in a collision state
        if len(set(start_s)) < N:
            self.collision_likelihood = 1.0
            return

        mu_safe[start_s] = 1.0

        for t in range(self.T):
            mu_safe_next = {}
            for s, prob in mu_safe.items():
                if prob < 1e-12:
                    continue
                # Product of each player's transition under their policy
                hats_list = [
                    self.reachable_set[(s[p], int(self.pols[p][s[p], t]))]
                    for p in range(N)
                ]
                for hat_s in ut.cartesian_product(hats_list):
                    # Y indicator: no collision in next state
                    if len(set(hat_s)) < N:
                        continue
                    trans_prob = np.prod([
                        self.Ps[p][hat_s[p], s[p], int(self.pols[p][s[p], t])]
                        for p in range(N)
                    ])
                    if trans_prob > 1e-14:
                        mu_safe_next[hat_s] = mu_safe_next.get(hat_s, 0.0) + prob * trans_prob
            mu_safe = mu_safe_next

        prob_no_collision     = sum(mu_safe.values())
        self.collision_likelihood = 1.0 - prob_no_collision
        self._mu_safe_final   = mu_safe  # store for reach reduction

    def get_reach_reduction(self):
        """
        Reach reduction = E[prod_j X_j(s^T_j) | tau_j ~ h_j(pi*_j)]
                        / prod_j E[X_j(s^T_j) | tau_j ~ h_j(pi^single_j)]

        Equation (28) of the paper: ratio of the joint reach probability under
        the Nash policy to the product of individual optimal reach probabilities.
        Numerator  — joint probability all agents reach targets collision-free.
        Denominator — product of single-agent optimal reach probabilities.

        Requires get_collision_likelihood() to have been called first
        (uses self._mu_safe_final).
        """
        if not hasattr(self, '_mu_safe_final'):
            self.get_collision_likelihood()

        N = self.player_num

        # Numerator: sum of safe occupancy mass at states where all agents at targets
        joint_reach_prob = sum(
            prob for s, prob in self._mu_safe_final.items()
            if all(s[p] == self.targ_raw_inds[p] for p in range(N))
        )

        # Denominator: product of single-agent optimal reach probabilities
        # Solve single-agent reachability MDP for each player independently
        indep_reach_prod = 1.0
        for p in range(N):
            V_indep, _ = vi.finite_reachability(
                self.Ps[p], self.T, self.targ_raw_inds[p]
            )
            indep_reach_prod *= V_indep[self.x_0s[p], 0]

        if indep_reach_prod > 0:
            self.reach_reduction = joint_reach_prob / indep_reach_prod
        else:
            self.reach_reduction = 0.0

    def get_all_metrics(self):
        """Convenience method: compute all three paper metrics in sequence."""
        self.get_potential()
        self.get_collision_likelihood()
        self.get_reach_reduction()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def highest_prob_trajectory(self):
        """
        Greedily follows the most probable next state at each timestep
        for each player under their converged local policy.
        Stores result in self.traj as a list of joint state tuples.
        """
        current_s = tuple(self.x_0s)
        self.traj = [current_s]

        for t in range(self.T):
            next_s = []
            for p in range(self.player_num):
                a = int(self.pols[p][current_s[p], t])
                candidates = [
                    (self.Ps[p][sp, current_s[p], a], sp)
                    for sp in self.reachable_set[(current_s[p], a)]
                ]
                best_sp = max(candidates, key=lambda x: x[0])[1]
                next_s.append(best_sp)
            current_s = tuple(next_s)
            self.traj.append(current_s)

    def plot_occupancy_grid(self):
        """
        Plots grid with start/target markers and overlaid greedy trajectories.
        """
        if not hasattr(self, 'traj'):
            self.highest_prob_trajectory()

        data = np.zeros((self.Rows, self.Columns))
        for t_idx in self.targ_raw_inds:
            r, c = divmod(t_idx, self.Columns)
            data[r, c] = 1
        for s_idx in self.x_0s:
            r, c = divmod(s_idx, self.Columns)
            data[r, c] = 2

        my_cmap = ListedColormap(["#f0f0f0", "#2ecc71", "#3498db"])
        plt.figure(figsize=(self.Columns * 1.2, self.Rows))
        sns.set_style("white")
        annot = np.arange(self.Rows * self.Columns).reshape(self.Rows, self.Columns)
        ax = sns.heatmap(data, annot=annot, fmt="d", cmap=my_cmap,
                         cbar=False, linewidths=2, linecolor='black', square=True)

        path_colors = ['#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
        for p in range(self.player_num):
            p_indices = [step[p] for step in self.traj]
            coords    = [divmod(idx, self.Columns) for idx in p_indices]
            rows_c    = [c[0] + 0.5 for c in coords]
            cols_c    = [c[1] + 0.5 for c in coords]
            ax.plot(cols_c, rows_c,
                    color=path_colors[p % len(path_colors)],
                    linewidth=3, marker='o', markersize=6,
                    label=f'P{p} path', alpha=0.8, zorder=5)

        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor='#2ecc71', label='Target'),
                      Patch(facecolor='#3498db', label='Start')]
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=legend_els + handles,
                  bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"Local Policy Path: {self.Rows}x{self.Columns} grid")
        plt.tight_layout()
        plt.show()
        
        
        
        
class Local_MDP_aug:

    def __init__(self, Rows, Columns, T, player_num):
        self.Rows = Rows
        self.Columns = Columns
        self.T = T
        self.player_num = player_num

    # ------------------------------------------------------------------
    # Problem setup
    # ------------------------------------------------------------------

    def create_MDP(self, action_entropy, targs_x0s):
    
        #Build individual transition models and extract targets / initial states.
    
        print("Starting local MDP construction...")
        time_start = time.time()

        # Individual transition tensors P[s', s, a] for each player
        self.Ps = [
            mdp.transitions(self.Rows, self.Columns, p=action_entropy, with_stay=True)
            for _ in range(self.player_num)
        ]

        S, _, A = self.Ps[0].shape
        self.S = S
        self.A = A

        # targs_x0s layout: first player_num entries = targets,
        #                   next  player_num entries = initial states
        self.targ_raw_inds = targs_x0s[:self.player_num]
        self.x_0s          = targs_x0s[self.player_num:]

        print("-" * 40)
        print(f"Target nodes:  {self.targ_raw_inds}")
        print(f"Starting nodes:{self.x_0s}")

        # Make target states absorbing sinks (action index 4 = stay)
        for p in range(self.player_num):
            self.Ps[p][:, self.targ_raw_inds[p], 4] = 0.0
            self.Ps[p][self.targ_raw_inds[p], self.targ_raw_inds[p], 4] = 1.0

        # Reachable set: dict (s, a) -> list of reachable s'
        self.reachable_set = mdp.reachable_set(self.Ps[0])

        # All states reachable from s under any action — used in best response
        self.all_reachable = {s: set() for s in range(S)}
        for s in range(S):
            for a in range(A):
                self.all_reachable[s].update(self.reachable_set[(s, a)])

        time_end = time.time()
        print(f"Time to construct local MDP: {np.round(time_end - time_start, 2)} seconds.")

    # ------------------------------------------------------------------
    # Algorithm 2: Local Feedback Best Response for player i
    # ------------------------------------------------------------------

    def _best_response(self, i, pols, rhos):
       
        #Computes player i's best response policy given fixed opponent policies.

        #Implements Algorithm 2 from the paper:
        #  - Forward pass: compute opponent occupancy measures rho^t_{-i}
        #  - Backward pass: compute W^t_i(s_i) — player i's marginalized
        #    value function — via the multiplicative DP over the joint space,
        #    averaging out opponents with their occupancy measures.

        
        #W_i is an (S, T+1) array — player i's local value function
        #pi_i is an (S, T)   array — player i's best response policy
      
        N     = self.player_num
        S     = self.S
        T     = self.T
        N_opp = [j for j in range(N) if j != i]

        # propagate each opponent's occupancy forward using their current policy.
        # rhos[j] is already computed externally and passed in.

        # Backward pass: multiplicative DP lines 11-22
        # V^T_pi(s) = prod_j X_j(s^T_j) * prod_{i,j} Y_ij(s_i, s_j)
        # W^t_i(s_i) = sum_{s_{-i}} rho^t_{-i}(s_{-i}) * V^t_pi(s_i, s_{-i})

        # Terminal joint value function This is V^T_pi from equation (20) in the paper.
        v_ind  = 0
        v_prev = 1
        V = [{}, {}]

        S_list = list(range(S))

        # Initialize V^T: all agents at their respective targets, no collision
        for s in permutations(S_list, N):
            at_targets   = all(s[p] == self.targ_raw_inds[p] for p in range(N))
            # permutations already guarantees no repeated states (Y=1 satisfied)
            if at_targets:
                V[v_ind][s] = 2.0
            else:
                V[v_ind][s] = 1.0

        # Build tau[t][j]: joint (s'_j, s_j) -> probability mass
        # = P_j(s'_j | s_j, pi_j(s_j)) * rho_j(s_j, t)
        # This encodes the weighted two-step occupancy for each opponent.
        tau = [{j: {} for j in N_opp} for _ in range(T)]
        for t in range(T - 1, -1, -1):
            for j in N_opp:
                for sj in S_list:
                    if rhos[j][sj, t] > 1e-10:
                        aj = int(pols[j][sj, t])
                        for hat_sj in self.reachable_set[(sj, aj)]:
                            key = (hat_sj, sj)
                            tau[t][j][key] = (
                                self.Ps[j][hat_sj, sj, aj] * rhos[j][sj, t]
                            )

        # W_i(s_i, t): player i's marginalized value function over individual states
        # Shape: (S, T+1)
        W_i  = np.zeros((S, T + 1))
        pi_i = np.zeros((S, T), dtype=int)

        # Terminal W^T_i(s_i) = sum_{s_{-i}} rho^T_{-i}(s_{-i}) * V^T(s_i, s_{-i})
        # (line 12 of Algorithm 2)
        for s in V[v_ind].keys():
            # s is a full joint state tuple; s[i] is player i's component
            opp_rho_prod = np.prod([rhos[j][s[j], T] for j in N_opp])
            W_i[s[i], T] += opp_rho_prod * V[v_ind][s]

        # Backward induction: t = T-1 down to 0
        for t in range(T - 1, -1, -1):
            eps   = (1e-2) ** (0.75 * t + 3)
            v_ind  = (v_ind  + 1) % 2
            v_prev = (v_ind  + 1) % 2

            # Build new V^t from V^{t+1} (lines 17-21 of Algorithm 2)
            opp_transition_list = [tau[t][j].keys() for j in N_opp]

            for si in S_list:
                # eV[hat_si] accumulates the expected future value for player i
                # transitioning to hat_si, averaged over opponents' joint transitions
                eV = {hat_si: 0.0 for hat_si in self.all_reachable[si]}

                # Insert player i's possible transitions at position i in the list
                opp_transition_list.insert(
                    i, [(hat_si, si) for hat_si in self.all_reachable[si]]
                )

                # Iterate over all joint next-state combinations (s', s) pairs
                for hat_s_s in ut.cartesian_product(opp_transition_list):
                    hat_s = tuple(hss[0] for hss in hat_s_s)

                    # Y indicator: no collision in next joint state
                    no_collision = len(set(hat_s)) == N

                    if no_collision and hat_s in V[v_prev]:
                        # Opponent joint density: prod_{j != i} tau[t][j][(hat_sj, sj)]
                        opp_density = np.prod([
                            tau[t][j][hat_s_s[j if j < i else j][0],
                                      hat_s_s[j if j < i else j][1]]
                            if False else  # use dict key directly
                            tau[t][j][hat_s_s[j if j < i else j]]
                            for j in N_opp
                        ])
                        # indexing:
                        opp_density = 1.0
                        for j in N_opp:
                            idx = j if j < i else j  # position in hat_s_s after insert
                            opp_density *= tau[t][j][hat_s_s[idx]]

                        if opp_density >= 1e-12:
                            eV[hat_s_s[i][0]] += opp_density * V[v_prev][hat_s]

                opp_transition_list.pop(i)

                # Q[a] = sum_{s'_i} P_i(s'_i | s_i, a) * eV[s'_i]
                # (the argmax over Q gives pi*_i(s_i, t) — equation 24)
                Q = [
                    np.sum([
                        self.Ps[i][hat_si, si, a] * eV[hat_si]
                        for hat_si in self.reachable_set[(si, a)]
                    ])
                    for a in range(self.A)
                ]
                pi_i[si, t] = int(np.argmax(Q))

            # Update V^t for all collision-free joint states s
            # V^t(s) = Y(s) * sum_{s'} prod_j P_j(s'_j|s_j,pi_j) * V^{t+1}(s')
            for s in permutations(S_list, N):
                prod_rho = np.prod([rhos[j][s[j], t] for j in N_opp])
                if prod_rho <= eps:
                    if s in V[v_ind]:
                        del V[v_ind][s]
                    continue

                hats_list     = [self.reachable_set[(s[p], int(pols[p][s[p], t]))]
                                 for p in range(N)]
                reachable_hats = ut.cartesian_product(hats_list)
                valid_hats     = V[v_prev].keys() & reachable_hats

                V_s = sum(
                    V[v_prev][hat_s] * np.prod([
                        self.Ps[p][hat_s[p], s[p], int(pols[p][s[p], t])]
                        for p in range(N)
                    ])
                    for hat_s in valid_hats
                )

                if V_s > eps:
                    V[v_ind][s] = V_s
                elif s in V[v_ind]:
                    del V[v_ind][s]

            # W^t_i(s_i) = max_{a_i} sum_{s'_i} P_i(s'_i|s_i,a_i) * W^{t+1}_i(s'_i)
            # Already stored implicitly via pi_i above.
            # Store W^t_i for metric computation.
            for si in S_list:
                a_star = pi_i[si, t]
                W_i[si, t] = np.sum([
                    self.Ps[i][hat_si, si, a_star] * W_i[hat_si, t + 1]
                    for hat_si in self.reachable_set[(si, a_star)]
                ])

        V[0].clear()
        V[1].clear()
        return W_i, pi_i

    # ------------------------------------------------------------------
    # Algorithm 3: Iterative Best Response -> Nash equilibrium
    # ------------------------------------------------------------------

    def local_value_iteration(self, BR_iter=10, tol=1e-5):
        
        #Implements Algorithm 3: Iterative Best Response.

        # Each agent computes its best response in until
        # the potential value converges
         
        # run this until it returns a deterministic Nash equilibrium
        # in the local feedback policy space.

        
        print("-" * 40)
        print("Starting iterative best response (Algorithm 3)...")
        time_start = time.time()

        S = self.S
        T = self.T
        N = self.player_num

        # Initialize policies: flat (stay) policy — action 4
        self.pols = [np.full((S, T), 4, dtype=int) for _ in range(N)]

        # Initialize occupancy measures from flat policy
        self.rhos = [
            mdp.occupancy(self.pols[p], self.Ps[p], self.x_0s[p])
            for p in range(N)
        ]

        # Store W functions for each player (used in metric computation)
        self.Ws = [np.zeros((S, T + 1)) for _ in range(N)]

        prev_potential = -np.inf

        for k in range(BR_iter * N):
            i = k % N
            print(f"\nBR iteration {k // N + 1}, updating player {i}...")

            W_i, pi_i = self._best_response(i, self.pols, self.rhos)

            self.Ws[i]   = W_i
            self.pols[i] = pi_i
            self.rhos[i] = mdp.occupancy(self.pols[i], self.Ps[i], self.x_0s[i])

            # Check convergence every full round (all N players updated)
            if (k + 1) % N == 0:
                current_potential = self._compute_potential_from_W()
                delta = abs(current_potential - prev_potential)
                print(f"  Potential = {np.round(current_potential, 8)}, delta = {np.round(delta, 8)}")
                if delta < tol:
                    print(f"  Converged after {k // N + 1} full rounds.")
                    break
                prev_potential = current_potential

        time_end = time.time()
        self.time_elapsed = np.round(time_end - time_start, 2)
        print(f"\nTime for local value iteration: {self.time_elapsed} seconds.")

    # ------------------------------------------------------------------
    # Internal potential computation (used for convergence check)
    # ------------------------------------------------------------------

    def _compute_potential_from_W(self):
        
        #F(pi) = sum_s prod_i rho^0_i(s_i) * V^0_pi(s)
        #Approximated from W_i values at t=0 weighted by initial occupancy.
        #Uses the joint value function structure from Proposition 1.
        
        S = self.S
        N = self.player_num

        # Build sparse V^0 from current policies
        V = {}
        S_list = list(range(S))

        # Terminal V^T: targets, no collision
        for s in permutations(S_list, N):
            if all(s[p] == self.targ_raw_inds[p] for p in range(N)):
                V[s] = 2.0
            else:
                V[s] = 1.0

        for t in range(self.T - 1, -1, -1):
            V_new = {}
            for s in permutations(S_list, N):
                hats_list      = [self.reachable_set[(s[p], int(self.pols[p][s[p], t]))]
                                  for p in range(N)]
                reachable_hats = ut.cartesian_product(hats_list)
                valid_hats     = V.keys() & reachable_hats
                V_s = sum(
                    V[hat_s] * np.prod([
                        self.Ps[p][hat_s[p], s[p], int(self.pols[p][s[p], t])]
                        for p in range(N)
                    ])
                    for hat_s in valid_hats
                )
                if V_s > 1e-10:
                    V_new[s] = V_s
            V = V_new

        total_V = sum(
            V[s] * np.prod([self.rhos[p][s[p], 0] for p in range(N)])
            for s in V.keys()
        )
        return total_V

    # ------------------------------------------------------------------
    # Metric computation (paper definitions)
    # ------------------------------------------------------------------

    def get_potential(self):

        S      = self.S
        N      = self.player_num
        S_list = list(range(S))

        # Rebuild V^0 under converged policies using multiplicative DP
        V = {}
        for s in permutations(S_list, N):
            if all(s[p] == self.targ_raw_inds[p] for p in range(N)):
                V[s] = 1.0
             
        for t in range(self.T - 1, -1, -1):
            V_new = {}
            for s in permutations(S_list, N):
                hats_list      = [self.reachable_set[(s[p], int(self.pols[p][s[p], t]))]
                                  for p in range(N)]
                reachable_hats = ut.cartesian_product(hats_list)
                valid_hats     = V.keys() & reachable_hats
                V_s = sum(
                    V[hat_s] * np.prod([
                        self.Ps[p][hat_s[p], s[p], int(self.pols[p][s[p], t])]
                        for p in range(N)
                    ])
                    for hat_s in valid_hats
                )
                if V_s > 1e-10:
                    V_new[s] = V_s
            V = V_new

        start_s = tuple(self.x_0s)
        self.potential_value = V.get(start_s, 0.0)

    def get_collision_likelihood(self):
        
       # Collision likelihood = E[1 - prod_{t=0}^{T} prod_{i,j} Y_ij(s^t_i, s^t_j)]
                                #| tau_j ~ h_j(pi*_j)]

        #Equation (27) of the paper: probability that at least one collision
        #occurs at any timestep along the trajectory.

        #Computed via forward occupancy propagation over collision-free states only,
        #then collision_likelihood = 1 - P(no collision at any timestep).
        
        S      = self.S
        N      = self.player_num


        # mu_safe[(s, t)] = probability of being in collision-free joint state s at t
        mu_safe = {}
        start_s = tuple(self.x_0s)

        # Check if agents start in a collision state
        if len(set(start_s)) < N:
            self.collision_likelihood = 1.0
            return

        mu_safe[start_s] = 1.0

        for t in range(self.T):
            mu_safe_next = {}
            for s, prob in mu_safe.items():
                if prob < 1e-12:
                    continue
                # Product of each player's transition under their policy
                hats_list = [
                    self.reachable_set[(s[p], int(self.pols[p][s[p], t]))]
                    for p in range(N)
                ]
                for hat_s in ut.cartesian_product(hats_list):
                    # Y indicator: no collision in next state
                    if len(set(hat_s)) < N:
                        continue
                    trans_prob = np.prod([
                        self.Ps[p][hat_s[p], s[p], int(self.pols[p][s[p], t])]
                        for p in range(N)
                    ])
                    if trans_prob > 1e-14:
                        mu_safe_next[hat_s] = mu_safe_next.get(hat_s, 0.0) + prob * trans_prob
            mu_safe = mu_safe_next

        prob_no_collision     = sum(mu_safe.values())
        self.collision_likelihood = 1.0 - prob_no_collision
        self._mu_safe_final   = mu_safe  # store for reach reduction

    def get_reach_reduction(self):
        # Reach reduction = E[prod_j X_j(s^T_j) | tau_j ~ h_j(pi*_j)]
        #                / prod_j E[X_j(s^T_j) | tau_j ~ h_j(pi^single_j)]

        # Equation (28) of the paper: ratio of the joint reach probability under
        # the Nash policy to the product of individual optimal reach probabilities.
        # Numerator  — joint probability all agents reach targets collision-free.
        # Denominator — product of single-agent optimal reach probabilities.

        #Requires get_collision_likelihood() to have been called first
        #(uses self._mu_safe_final).
        
        # if not hasattr(self, '_mu_safe_final'):
        #     self.get_collision_likelihood()

        N = self.player_num

        # Numerator: sum of safe occupancy mass at states where all agents at targets
        joint_reach_prob = sum(
            prob for s, prob in self._mu_safe_final.items()
            if all(s[p] == self.targ_raw_inds[p] for p in range(N))
        )

        # Denominator: product of single-agent optimal reach probabilities
        # Solve single-agent reachability MDP for each player independently
        indep_reach_prod = 1.0
        for p in range(N):
            V_indep, _ = vi.finite_reachability(
                self.Ps[p], self.T, self.targ_raw_inds[p]
            )
            indep_reach_prod *= V_indep[self.x_0s[p], 0]

        if indep_reach_prod > 0:
            self.reach_reduction = joint_reach_prob / indep_reach_prod
        else:
            self.reach_reduction = 0.0

    def get_all_metrics(self):
        #compute all three paper metrics in sequence.
        self.get_potential()
        self.get_collision_likelihood()
        self.get_reach_reduction()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def highest_prob_trajectory(self):
        #follows the most probable next state at each timestep
        #for each player under their converged local policy.
        #Stores result in self.traj as a list of joint state tuples.
        
        current_s = tuple(self.x_0s)
        self.traj = [current_s]

        for t in range(self.T):
            next_s = []
            for p in range(self.player_num):
                a = int(self.pols[p][current_s[p], t])
                candidates = [
                    (self.Ps[p][sp, current_s[p], a], sp)
                    for sp in self.reachable_set[(current_s[p], a)]
                ]
                best_sp = max(candidates, key=lambda x: x[0])[1]
                next_s.append(best_sp)
            current_s = tuple(next_s)
            self.traj.append(current_s)

    def plot_occupancy_grid(self):
        
        #Plots grid with start/target markers and overlaid greedy trajectories.
        
        if not hasattr(self, 'traj'):
            self.highest_prob_trajectory()

        data = np.zeros((self.Rows, self.Columns))
        for t_idx in self.targ_raw_inds:
            r, c = divmod(t_idx, self.Columns)
            data[r, c] = 1
        for s_idx in self.x_0s:
            r, c = divmod(s_idx, self.Columns)
            data[r, c] = 2

        my_cmap = ListedColormap(["#f0f0f0", "#2ecc71", "#3498db"])
        plt.figure(figsize=(self.Columns * 1.2, self.Rows))
        sns.set_style("white")
        annot = np.arange(self.Rows * self.Columns).reshape(self.Rows, self.Columns)
        ax = sns.heatmap(data, annot=annot, fmt="d", cmap=my_cmap,
                         cbar=False, linewidths=2, linecolor='black', square=True)

        path_colors = ['#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
        for p in range(self.player_num):
            p_indices = [step[p] for step in self.traj]
            coords    = [divmod(idx, self.Columns) for idx in p_indices]
            rows_c    = [c[0] + 0.5 for c in coords]
            cols_c    = [c[1] + 0.5 for c in coords]
            ax.plot(cols_c, rows_c,
                    color=path_colors[p % len(path_colors)],
                    linewidth=3, marker='o', markersize=6,
                    label=f'P{p} path', alpha=0.8, zorder=5)

        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor='#2ecc71', label='Target'),
                      Patch(facecolor='#3498db', label='Start')]
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=legend_els + handles,
                  bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"Local Policy Path: {self.Rows}x{self.Columns} grid")
        plt.tight_layout()
        plt.show()