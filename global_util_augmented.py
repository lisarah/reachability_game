# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:11:45 2026

@author: adamc
"""


import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import MDP_algorithms.mdp_state_action as mdp
import MDP_algorithms.value_iteration as vi
import seaborn as sns
import time
from itertools import product
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import tracemalloc



class Global_MDP_aug:
    
    def __init__(self,Rows,Columns,T,player_num,y_val=1.0):
        self.Rows = Rows
        self.Columns = Columns
        self.T = T
        self.player_num = player_num
        self.Y = y_val
        
    # Problem setup and solution
        
    def create_joint_MDP(self,action_entropy,targs_x0s):
        
        print("Starting joint MDP construction...")
        
        time_start = time.time()
        
        # Returns an array of P that corresponds to s',s, and the action for each player
        self.Ps = [mdp.transitions(self.Rows, self.Columns, p=action_entropy, with_stay=True) for p in range(self.player_num)]
        
        # The target states are the second set of random numbers
        self.targ_raw_inds = targs_x0s[:self.player_num]    
        
        # The initial indices are the first set of random numbers       
        self.x_0s = targs_x0s[self.player_num:]
        
        # turn targets into sinks
        for p in range(self.player_num):
            S_p = self.Ps[p].shape[0]
            
            # Absorbing at the target cell
            self.Ps[p][:, self.targ_raw_inds[p], 4] = np.zeros((S_p))
            self.Ps[p][self.targ_raw_inds[p], self.targ_raw_inds[p], 4] = 1.
        
        print("-"*40)
        print(f"Target nodes: {self.targ_raw_inds}")
        print(f"Starting nodes: {self.x_0s}")

        # Determines reachable sets for each player to avoid np.where inside the loop because that was taking way too long
        R_sets = [mdp.reachable_set(P) for P in self.Ps]
        
        # Get our state space dimension and action space dimension
        S, _, A = self.Ps[0].shape
        
        # To create the joint state and action space, I used this itertools function. Repeat is the number of times you call
        # product on the positional argument.
        self.joint_states = list(product(range(S), repeat=self.player_num))
        self.joint_actions = list(product(range(A), repeat=self.player_num))
        
        # I need a way to index the joint state so I used a dictionary to map between them
        self.state_to_idx = {s: i for i, s in enumerate(self.joint_states)}
        self.idx_to_state = {i: s for s, i in self.state_to_idx.items()}
        
        # Specify dimensions for the joint spaces
        self.S_joint = len(self.joint_states) # S^2
        self.A_joint = len(self.joint_actions) # A^2
        
        # Initialize the Ps from above but in the joint space
        # Structured as (next_state, current_state, action)
        # Has dimension (S^2,S^2,A^2)
        # Use dictionary instead of np.zeros to reduce memory usage
        self.P_joint = {}
        
        # Now loop through every possible joint state 
        for s_idx, s_tuple in enumerate(self.joint_states):
            #print(s_idx,s_tuple)
            
            # In each of these states, loop through the possible actions. Might have to change if action space is dependent on state.
            for a_idx, a_tuple in enumerate(self.joint_actions):
                #print(a_idx,a_tuple)
                
                # next_probs will be used to store the indices corresponding to the next state which each player can land
                # Optimization: Use reachable sets
                next_probs = [R_sets[p][(s_tuple[p], a_tuple[p])] for p in range(self.player_num)]
                
                # Now compute the joint probability of landing in a next state
                # First, we have to evaluate each outcome of the nonzero probabilities. That is what product(*next_prob) does.
                # For example, if next_prob is [[1,2],[3,4]], then we would loop through [1,3],[1,4],[2,3], and [2,4]
                for s_next_tuple in product(*next_probs):
                    # Initialize our probability for the joint state
                    prob = 1.0
                    
                    # For each player, we calculate the probability of landing in a joint state by multiplying the probability of landing in each individual state
                    for p in range(self.player_num):
                        prob *= self.Ps[p][s_next_tuple[p], s_tuple[p], a_tuple[p]]
                    
                    # If this configuration is possible...
                    if prob > 1e-12:
                        
                        # ... Then we take our mapping from the tuple s_next_tuple to the index 
                        sp_idx = self.state_to_idx[s_next_tuple]
                        
                        # We add prob to the final joint probability distribution at a given action, state, and next state.
                        # Optimization: Sparse dictionary assignment
                        self.P_joint[(sp_idx, s_idx, a_idx)] = self.P_joint.get((sp_idx, s_idx, a_idx), 0.0) + prob
        
        time_end = time.time()
        
        # Check if the state transitions sum to 1 for all possible next states
        # Calculate sums from dictionary values
        column_sums = {}
        
        # Looping through s', s, and a in the sparse P_joint...
        for (sp, s, a), p in self.P_joint.items():
            
            # ... we get the value corresponding to a state and action for every possible s'. If the index doesn't exist, it returns zero.
            # This was returning a ValueError unless I added the zero 
            column_sums[(s, a)] = column_sums.get((s, a), 0.0) + p
        
        
        print(f"Time to construct sparse joint P: {np.round(time_end - time_start,2)} seconds.")
        
    def global_value_iteration(self):
        
        #----------------------------------------------------------------------
        # THIRD CHUNK OF CODE: Global policy best response
        #----------------------------------------------------------------------
        
        print("-"*40)
        print('Starting individual player best response.')
        
        time_start = time.time()
        
        # Intialize value function V for each joint state per timestep
        self.V = np.zeros((self.S_joint, self.T+1))
        
        # Construct V_T: Enforce X and Y at time T
        for s_idx, s_tuple in self.idx_to_state.items():
            
            # X_i: Check if all players are at their respective targets
            at_targets = all(s_tuple[p] == self.targ_raw_inds[p] for p in range(self.player_num))
            
            # If yes, 1 otherwise 0
            #X_prod = 1.0 if at_targets else 0.0

            # Y product: Check for collisions
            no_collisions = len(set(s_tuple)) == len(s_tuple)
            
            # If the set is not the same size as the tuple, then agents occupy the same state
            #Y_prod = 1.0 if no_collisions else 0.0


            # The Value function at the final time T is the product of these two conditions
            if not no_collisions:
                self.V[s_idx, self.T] = 0.0        
            elif at_targets:
                self.V[s_idx, self.T] = 2.0        
            else:
                self.V[s_idx, self.T] = 1.0  
                
            # Remove the intermediate X,Y and directly assign to V. Make it more readable for myself.

        
        # Now start backwards induction 
        

            # --------------------dictionary instead sparse-------------------------
        # Initialize global policy
        self.pi_i_global = np.zeros((self.S_joint, self.T), dtype=int) 
            # --------------------dictionary instead sparse-------------------------
            
            
        # Start at the second to last timestep and move backwards to the start
        for t in range(self.T - 1, -1, -1):
        
            print('Calculating for timestep', t,'...')    

            action_values = np.zeros(self.A_joint) # Looking for the best joint action
            
            # Loop through every combination of the joint space
            for s_idx in range(self.S_joint):
                
                # Define the state tuple coordinates
                s_tuple = self.idx_to_state[s_idx]
                
                # We need the product of Y for the current state (Line 6 of Algorithm 4)
                no_collisions = len(set(s_tuple)) == len(s_tuple)
                
                # Same logic as initialization
                #Y_prod = self.Y if no_collisions else 0.0
                Y_prod = 1.0 if no_collisions else 0.0
                
                # If Y_prod is 0.0, we can break and then assign assign value to zero and skip this             
    
                # Initialize joint action values because we want to pick the maximum of this array
                action_values.fill(0.0) # Reset the action values at every state
                
# ---------sum of prod and v in range sp in s_joint reachable set----------------
                # Looping through each possible action
                for a_idx in range(self.A_joint):
                    
                    # Initialize the expected value for a specific action
                    expected_val = 0.0
                    
                    # For every state we would land in...
                    for sp_idx in range(self.S_joint):
                        
                        # ... obtain the probability that taking this action lands in sp
                        prob = self.P_joint.get((sp_idx, s_idx, a_idx), 0.0)
 # ---------sum of prod and v in range sp in s_joint reachable set----------------
                        
            # --------------------Maybe rid of this-------------------------
                        # Only if the state is reachable...
                        if prob > 0:
                            
                            #  ...do we multiply the probability with V_{t+1} of being in that state
                            expected_val += prob * self.V[sp_idx, t+1]
                            
            # --------------------Maybe rid of this-------------------------
                                        
            
            
            # --------------------Maybe rid of this-------------------------
                    # Lastly, take the future value and multiply with the indicator Y
                    action_values[a_idx] = Y_prod * expected_val
            # --------------------Maybe rid of this-------------------------
                    
            # Stop computation here and start after.

            
                # Pick action with the highest value and store in the policy
                self.pi_i_global[s_idx, t] = np.argmax(action_values)
                
                
                # Store the action index that maximizes value function
                best_a_idx = self.pi_i_global[s_idx, t]
                
                # Initialize a container for the expected value of taking the best action
                v_sum = 0.0
                
                # For each possible state you can land in given the best action...
                for sp_idx in range(self.S_joint):
                    
                    # ... we take the probability of landing in sp from the joint probability
                    prob = self.P_joint.get((sp_idx, s_idx, best_a_idx), 0.0)
                    
                    # ... and multiply it by the value function at sp, adding it to v_sum to get our expected value
                    v_sum += prob * self.V[sp_idx, t+1]
                
                
                # Threshold on self.V (1e-1)**(t+3)
                # Make V function dictionary -> t is a list, s idx is a dict
                # instead of iterating over joint iterate over keys.
                # elif -> if s velow V threshold is for some reason in V get rid of it
                # Break these down from getting actions to getting Vs: for a in a_idx to v_sum=0.0, and for sp_idx to end

     
                self.V[s_idx, t] = Y_prod * v_sum
        
        
        time_end = time.time()
        self.time_elapsed = np.round(time_end-time_start,2)
        print('Time to run the global policy:',self.time_elapsed,'seconds.')
        

    def get_potential(self):

                
        # Initialize the occupancy measure as an array of states X timesteps.
        # Each entry represents the probability of the joint system being in state s at time t
        mu = np.zeros((self.S_joint, self.T + 1))
        
        # Pulls the index of the starting states
        start_idx = self.state_to_idx[tuple(self.x_0s)]
        
        # Assign an occupancy measure of 1.0 to the starting states as it is deterministic
        mu[start_idx, 0] = 1.0
    
        # For all timesteps from 0 to T-1...
        for t in range(self.T):
            
            # ... first determine states that have probability mass to go faster
            active_states = np.where(mu[:, t] > 1e-12)[0]
            
            # for each active state...
            for s_idx in active_states:
                
                # get the tuple of states
                s_tuple = self.idx_to_state[s_idx]
                
                # perform set check for the states. If there is a collision...
                if len(set(s_tuple)) < self.player_num:
                    
                    # ... do not add the probability to the potential
                    continue
                
                # otherwise, we pull the action that the policy determines
                a_idx = self.pi_i_global[s_idx, t]
                
                # look at reachable states under this particular action
                for (sp_idx, s_look, a_look), prob in self.P_joint.items():
                    
                    # If it is the current state and action...
                    if s_look == s_idx and a_look == a_idx:
                        
                        # ...look at the next state and check the set condition again 
                        sp_tuple = self.idx_to_state[sp_idx]
                        
                        # set condition for s'
                        if len(set(sp_tuple)) == self.player_num:
                            
                            # only if they are collision free do we add the probability to the occupancy measure
                            mu[sp_idx, t+1] += mu[s_idx, t] * prob
    
        # We have completed the forward pass. Go through the joint states
        
        # Initialize potential
        potential = 0.0
        
        # For each jjoint state...
        for s_idx in range(self.S_joint):
            
            # Convert to tuples
            s_tuple = self.idx_to_state[s_idx]
            
            # Set condition for target reached
            at_targets = all(s_tuple[p] == self.targ_raw_inds[p] for p in range(self.player_num))
            
            # redundant but ensure that there are no collisions
            no_collision = len(set(s_tuple)) == self.player_num
    
            # If both of these are met...
            if at_targets and no_collision:
                
                # Add the probability mass to the potential
                potential += mu[s_idx, self.T]
    
        self.potential = potential

    def get_collision_likelihood(self):
        self.start_idx = self.state_to_idx[tuple(self.x_0s)]
        
        # Check initial safety
        if len(set(self.idx_to_state[self.start_idx])) < self.player_num:
            self.collision_likelihood = 1.0
            return
    
        self.mu_safe = np.zeros((self.S_joint, self.T + 1))
        self.mu_safe[self.start_idx, 0] = 1.0
    
        for t in range(self.T):
            # iterate over states that actually have probability mass
            active_states = np.where(self.mu_safe[:, t] > 1e-12)[0]
            
            for s_idx in active_states:
                a_idx = self.pi_i_global[s_idx, t]
                
                # Find all possible next states from P_joint
                # (Filtering P_joint for current s_idx and a_idx)
                for (sp_idx, s_look, a_look), prob in self.P_joint.items():
                    if s_look == s_idx and a_look == a_idx:
                        sp_tuple = self.idx_to_state[sp_idx]
                        
                        # Y indicator: 1 if safe, 0 if collision
                        if len(set(sp_tuple)) == len(sp_tuple):
                            self.mu_safe[sp_idx, t+1] += self.mu_safe[s_idx, t] * prob
    

        prob_no_collision = np.sum(self.mu_safe[:, self.T])
        self.collision_likelihood = 1.0 - prob_no_collision

    def get_reach_reduction(self):
        
        # Initialize the starting index and potential value


        
        self.joint_reach_prob = 0.0
        for s_idx in range(self.S_joint):
            s_tuple = self.idx_to_state[s_idx]
            at_goal = all(s_tuple[j] == self.targ_raw_inds[j] for j in range(self.player_num))
            if at_goal:
                self.joint_reach_prob += self.mu_safe[s_idx, self.T]

        # --- 4. Reach Reduction Denominator (Eq 32 Denominator) ---
        # We need the product of individual probabilities of reaching targets.
        # We solve a simple local Reachability MDP for each player to get pi_star.
        indep_reach_probs_product = 1.0
        for p in range(self.player_num):
            # Calculate individual reach probability for player p using Ps[p]
            # This uses the shortest path / optimal reachability value at t=0
            V_indep, _ = vi.finite_reachability(self.Ps[p], self.T, self.targ_raw_inds[p])
            indep_reach_probs_product *= V_indep[self.x_0s[p], 0]

        # Handle division by zero just in case
        if indep_reach_probs_product > 0:
            self.reach_reduction = self.joint_reach_prob / indep_reach_probs_product
        else:
            self.reach_reduction = 0.0

    def get_global_metrics(self):
        
        self.get_potential()
        self.get_collision_likelihood()
        self.get_reach_reduction()

    # Visualization

    def plot_occupancy_grid(self):
        # 1. Setup Grid Data
        start_states = self.x_0s
        target_states = self.targ_raw_inds
        data = np.zeros((self.Rows, self.Columns))
        
        for t_idx in target_states:
            r, c = divmod(t_idx, self.Columns)
            data[r, c] = 1
        for s_idx in start_states:
            r, c = divmod(s_idx, self.Columns)
            data[r, c] = 2

        # 2. Setup Aesthetics
        my_colors = ["#f0f0f0", "#2ecc71", "#3498db"]
        my_cmap = ListedColormap(my_colors)
        plt.figure(figsize=(self.Columns*1.2, self.Rows))
        sns.set_style("white")
        
        annot_matrix = np.arange(self.Rows * self.Columns).reshape(self.Rows, self.Columns)
        
        # 3. Draw Base Heatmap
        ax = sns.heatmap(
            data, annot=annot_matrix, fmt="d", cmap=my_cmap, 
            cbar=False, linewidths=2, linecolor='black', square=True
        )

        # 4. Draw Trajectory Lines from self.traj
        path_colors = ['#e74c3c', '#9b59b6'] # Red for P0, Purple for P1
        
        # Ensure self.traj exists before iterating
        if hasattr(self, 'traj') and self.traj is not None:
            for p in range(self.player_num):
                # Pull player p's indices from the stored joint trajectory
                p_indices = [step[p] for step in self.traj]
                
                coords = [divmod(idx, self.Columns) for idx in p_indices]
                rows = [c[0] + 0.5 for c in coords]
                cols = [c[1] + 0.5 for c in coords]
                
                ax.plot(cols, rows, color=path_colors[p % len(path_colors)], 
                        linewidth=3, marker='o', markersize=6, 
                        label=f'P{p} Optimal Path', alpha=0.8, zorder=5)
        else:
            print("Warning: self.traj not found. Run highest_prob_trajectory() first.")

        # 5. Legend and Finalize
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Target'),
            Patch(facecolor='#3498db', label='Start')
        ]
        # Merge patches with path labels
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=legend_elements + handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(f"Reachability MDP Path: {self.Rows}x{self.Columns}")
        plt.show()
        
    def highest_prob_trajectory(self):
        current_s_idx = self.state_to_idx[tuple(self.x_0s)]
        self.traj = []
    
        for t in range(self.T):
            state_tuple = self.idx_to_state[current_s_idx]
            self.traj.append(state_tuple)
            
            a_idx = self.pi_i_global[current_s_idx, t]
            
            # Filter P_joint for current (s, a) to find the most likely s'
            options = [(sp, p) for (sp, s, a), p in self.P_joint.items() 
                       if s == current_s_idx and a == a_idx]
            
            if options:
                # Append most probable next state
                current_s_idx = max(options, key=lambda x: x[1])[0]
            else:
                break
                
        # Append the final state at time T
        self.traj.append(self.idx_to_state[current_s_idx])

    def propagate_joint_occupancy(self):
        """
        Calculates the joint occupancy measure mu[s_joint, t] for the global policy.
        """
        # Initialize joint occupancy: (S_joint, T+1)
        self.mu_joint = np.zeros((self.S_joint, self.T + 1))
        
        # Initial state has probability 1.0
        start_idx = self.state_to_idx[tuple(self.x_0s)]
        self.mu_joint[start_idx, 0] = 1.0

        for t in range(self.T):
            # Only iterate over states that have probability mass
            active_states = np.where(self.mu_joint[:, t] > 1e-12)[0]
            
            for s_idx in active_states:
                # Get the action prescribed by the global policy
                a_idx = self.pi_i_global[s_idx, t]
                
                # Update next states based on transition probabilities
                # We iterate through P_joint for (s_idx, a_idx)
                for (sp_idx, s_look, a_look), prob in self.P_joint.items():
                    if s_look == s_idx and a_look == a_idx:
                        self.mu_joint[sp_idx, t+1] += self.mu_joint[s_idx, t] * prob
        
        print("Joint occupancy propagation complete.")


    def animate_occupancy(self, interval=600):
        """
        Creates a side-by-side heatmap animation of player occupancy.
        """
        import matplotlib.pyplot as plt


        # Check if occupancy has been calculated
        if not hasattr(self, 'mu_joint'):
            self.propagate_joint_occupancy()

        fig, axes = plt.subplots(1, self.player_num, figsize=(self.player_num * 4, 4))
        if self.player_num == 1:
            axes = [axes]

        ims = []
        for p in range(self.player_num):
            im = axes[p].imshow(np.zeros((self.Rows, self.Columns)), 
                                cmap='rocket', origin='upper', vmin=0, vmax=1)
            axes[p].set_title(f"Player {p} Occupancy")
            # Mark the targets on the grid for reference
            tr, tc = divmod(self.targ_raw_inds[p], self.Columns)
            axes[p].plot(tc, tr, 'wo', markersize=10, markeredgecolor='black', label='Target')
            ims.append(im)

        def update(t):
            marginal_grids = [np.zeros((self.Rows, self.Columns)) for _ in range(self.player_num)]
            
            # Marginalize joint state s_idx at time t
            for s_idx in range(self.S_joint):
                prob = self.mu_joint[s_idx, t]
                if prob > 1e-9:
                    s_tuple = self.idx_to_state[s_idx]
                    for p in range(self.player_num):
                        r, c = divmod(s_tuple[p], self.Columns)
                        marginal_grids[p][r, c] += prob
            
            for p in range(self.player_num):
                ims[p].set_data(marginal_grids[p])
            
            fig.suptitle(f"Global Policy Occupancy | Timestep t = {t}")
            return ims

        ani = FuncAnimation(fig, update, frames=self.T + 1, interval=interval, blit=False)
        plt.tight_layout()
        plt.show()
        return ani
