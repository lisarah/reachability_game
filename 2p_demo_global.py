# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:11:45 2026

@author: adamc
"""


import numpy as np
import visualization as vs
import util as ut
import matplotlib.pyplot as plt
import pandas as pd
import MDP_algorithms.mdp_state_action as mdp
import MDP_algorithms.value_iteration as vi
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import potential_games as game
import os
import time
from itertools import product


Columns = 4
Rows = 4
T = 10
player_num = 2

# This is not used
change_horizon = [(0.9, 5*i+1) for i in range(10)]

# A list of probabilities to determine how likely an agent is to move to its intended target cell 
# I don't think this is necessary. Also, shouldn't the 15 be T? It's overwritten

#change_entropy = [(0.1*(i+1), 15) for i in range(10)]

#change_entropy = [(0.1*(i+1), T) for i in range(10)]
change_entropy = [(0.1,T)]
#action_entropy = 0.7
# We are looping through entropy
#for action_entropy, T in change_entropy:



def global_metrics(V, pi_i_global, P_joint, Ps, x_0s, targ_raw_inds, T):

    
    # Initialize the starting index and potential value
    start_idx = state_to_idx[tuple(x_0s)]
    potential_value = V[start_idx, 0]

    # Propagate forward in time
    mu_safe = np.zeros((S_joint, T + 1))
    
    # Initial conditions start with a probability of 1
    mu_safe[start_idx, 0] = 1.0
    player_num = len(x_0s)

    # For each timestep from 0 to T...
    for t in range(T):
        
        # Initialize a container for the joint state space
        new_mu = np.zeros(S_joint)
        
        # Looping through each possible state...
        for s_idx in range(S_joint):
            
            # If the specific state is reachable...
            if mu_safe[s_idx, t] > 0:
                
                # Pull the optimal action from the policy at that state and time
                a_idx = pi_i_global[s_idx, t]
                
                # For each possible outcome S' of taking that action A at state S...
                for (sp_idx, s_lookup, a_lookup), prob in P_joint.items():
                    
                    
                    if s_lookup == s_idx and a_lookup == a_idx:
                    
                        sp_tuple = idx_to_state[sp_idx]
                        
                        if len(set(sp_tuple)) == len(sp_tuple): # is_sp_safe check
                            new_mu[sp_idx] += mu_safe[s_idx, t] * prob
        mu_safe[:, t+1] = new_mu

    # Calculate the probability of no collision by taking the total probability of landing safely in time T
    prob_never_collided = np.sum(mu_safe[:, T])
    
    # The probability of collision is 1 - no_collision
    collision_likelihood = 1.0 - prob_never_collided




    # This is not correct
    
    joint_reach_prob = 0.0
    for s_idx in range(S_joint):
        s_tuple = idx_to_state[s_idx]
        at_goal = all(s_tuple[j] == targ_raw_inds[j] for j in range(player_num))
        if at_goal:
            joint_reach_prob += mu_safe[s_idx, T]

    # --- 4. Reach Reduction Denominator (Eq 32 Denominator) ---
    # We need the product of individual probabilities of reaching targets.
    # We solve a simple local Reachability MDP for each player to get pi_star.
    indep_reach_probs_product = 1.0
    for p in range(player_num):
        # Calculate individual reach probability for player p using Ps[p]
        # This uses the shortest path / optimal reachability value at t=0
        V_indep, _ = vi.finite_reachability(Ps[p], T, targ_raw_inds[p])
        indep_reach_probs_product *= V_indep[x_0s[p], 0]

    # --- 5. Reach Reduction Ratio ---
    # Handle division by zero just in case
    if indep_reach_probs_product > 0:
        reach_reduction = joint_reach_prob / indep_reach_probs_product
    else:
        reach_reduction = 0.0

    return potential_value, collision_likelihood,reach_reduction


# Initialize list for all global results
all_global_results = []

for action_entropy, _ in change_entropy:

# FIRST CHUNK: CONVERT TO JOINT MDP

    print("Starting joint MDP construction...")
    
    time_start = time.time()
    
    # Returns an array of P that corresponds to s',s, and the action for each player
    Ps = [mdp.transitions(Rows, Columns, p=action_entropy, with_stay=True) for p in range(player_num)]
    
    
    
    #---------------------------------------------------------------------
    # Initialize the targets. If not random, this will have to change
    
    targs_x0s = np.random.choice(range(Columns*Rows), 
                                size=player_num*2, replace=False) 
    
    
    # The target indices are the second set of random numbers
    targ_raw_inds = targs_x0s[:player_num]    
    
    # The initial conditions are the first set of random numbers       
    x_0s = targs_x0s[player_num:]
    
    # turn targets into sinks
    for p in range(player_num):
        S_p = Ps[p].shape[0]
        
        # Absorbing at the target cell
        Ps[p][:, targ_raw_inds[p], 4] = np.zeros((S_p))
        Ps[p][targ_raw_inds[p], targ_raw_inds[p], 4] = 1.
    
    print("-"*40)
    print(f"Target nodes: {targ_raw_inds}")
    print(f"Starting nodes: {x_0s}")
    
    #---------------------------------------------------------------------
    
    
    
    #---------------------------------------------------------------------
    # Make this its own function: create joint mdp or something
    #---------------------------------------------------------------------
    
    
    # Determines reachable sets for each player to avoid np.where inside the loop because that was taking way too long
    R_sets = [mdp.reachable_set(P) for P in Ps]
    
    # Get our state space dimension and action space dimension
    S, _, A = Ps[0].shape
    
    # To create the joint state and action space, I used this itertools function. Repeat is the number of times you call
    # product on the positional argument.
    joint_states = list(product(range(S), repeat=player_num))
    joint_actions = list(product(range(A), repeat=player_num))
    
    # I need a way to index the joint state so I used a dictionary to map between them
    state_to_idx = {s: i for i, s in enumerate(joint_states)}
    idx_to_state = {i: s for s, i in state_to_idx.items()}
    
    # Specify dimensions for the joint spaces
    S_joint = len(joint_states) # S^2
    A_joint = len(joint_actions) # A^2
    
    # Initialize the Ps from above but in the joint space
    # Structured as (next_state, current_state, action)
    # Has dimension (S^2,S^2,A^2)
    # Use dictionary instead of np.zeros to reduce memory usage
    P_joint = {}
    
    # Now loop through every possible joint state 
    for s_idx, s_tuple in enumerate(joint_states):
        #print(s_idx,s_tuple)
        
        # In each of these states, loop through the possible actions. Might have to change if action space is dependent on state.
        for a_idx, a_tuple in enumerate(joint_actions):
            #print(a_idx,a_tuple)
            
            # next_probs will be used to store the indices corresponding to the next state which each player can land
            # Optimization: Use reachable sets
            next_probs = [R_sets[p][(s_tuple[p], a_tuple[p])] for p in range(player_num)]
            
            # Now compute the joint probability of landing in a next state
            # First, we have to evaluate each outcome of the nonzero probabilities. That is what product(*next_prob) does.
            # For example, if next_prob is [[1,2],[3,4]], then we would loop through [1,3],[1,4],[2,3], and [2,4]
            for s_next_tuple in product(*next_probs):
                # Initialize our probability for the joint state
                prob = 1.0
                
                # For each player, we calculate the probability of landing in a joint state by multiplying the probability of landing in each individual state
                for p in range(player_num):
                    prob *= Ps[p][s_next_tuple[p], s_tuple[p], a_tuple[p]]
                
                # If this configuration is possible...
                if prob > 1e-12:
                    
                    # ... Then we take our mapping from the tuple s_next_tuple to the index 
                    sp_idx = state_to_idx[s_next_tuple]
                    
                    # We add prob to the final joint probability distribution at a given action, state, and next state.
                    # Optimization: Sparse dictionary assignment
                    P_joint[(sp_idx, s_idx, a_idx)] = P_joint.get((sp_idx, s_idx, a_idx), 0.0) + prob
    
    time_end = time.time()
    
    # Check if the state transitions sum to 1 for all possible next states
    # Calculate sums from dictionary values
    column_sums = {}
    
    # Looping through s', s, and a in the sparse P_joint...
    for (sp, s, a), p in P_joint.items():
        
        # ... we get the value corresponding to a state and action for every possible s'. If the index doesn't exist, it returns zero.
        # This was returning a ValueError unless I added the zero 
        column_sums[(s, a)] = column_sums.get((s, a), 0.0) + p
    
    # P_joint should be column stochastic. Check if every single entry in (S_joint, A_joint) is close to 1.0
    is_valid = all(np.isclose(val, 1.0) for val in column_sums.values())
    
    #print(f"Columns sum to one: {is_valid}")
    #print(f"Shape of P (S'_joint,S_joint,A_joint): ({S_joint}, {S_joint}, {A_joint})")
    #print(f"Stored non-zero entries: {len(P_joint)} compared to {S_joint**2} possible entries")
    print(f"Time to construct sparse joint P: {np.round(time_end - time_start,2)} seconds.")
    
            #return P_joint, state_to_idx
    
    print(P_joint)
    #----------------------------------------------------------------------
    # End function right here
    #----------------------------------------------------------------------
    
    
    
    
    
    
    #----------------------------------------------------------------------
    # THIRD CHUNK OF CODE: Global policy best response
    #----------------------------------------------------------------------
    
    print("-"*40)
    print('Starting individual player best response.')
    
    time_start = time.time()
    
    # Intialize value function V for each joint state per timestep
    V = np.zeros((S_joint, T+1))
    
    # Construct V_T: Enforce X and Y at time T
    for s_idx, s_tuple in idx_to_state.items():
        
        # X_i: Check if all players are at their respective targets
        at_targets = all(s_tuple[p] == targ_raw_inds[p] for p in range(player_num))
        
        # If yes, 1 otherwise 0
        X_prod = 1.0 if at_targets else 0.0
        
        # Y product: Check for collisions
        no_collisions = len(set(s_tuple)) == len(s_tuple)
        
        # If the set is not the same size as the tuple, then agents occupy the same state
        Y_prod = 1.0 if no_collisions else 0.0
        
        # The Value function at the final time T is the product of these two conditions
        V[s_idx, T] = X_prod * Y_prod
    
    
    
    # Now start backwards induction 
    
    # Initialize global policy
    pi_i_global = np.zeros((S_joint, T), dtype=int) 
    
    # Start at the second to last timestep and move backwards to the start
    for t in range(T - 1, -1, -1):
    
        print('Calculating for timestep', t,'...')    
    
        # Loop through every combination of the joint space
        for s_idx in range(S_joint):
            
            # Define the state tuple coordinates
            s_tuple = idx_to_state[s_idx]
            
            # We need the product of Y for the current state (Line 6 of Algorithm 4)
            no_collisions = len(set(s_tuple)) == len(s_tuple)
            
            # Same logic as initialization
            Y_prod = 1.0 if no_collisions else 0.0
            
            # Initialize joint action values because we want to pick the maximum of this array
            action_values = np.zeros(A_joint) # Looking for the best joint action
            
            # Looping through each possible action
            for a_idx in range(A_joint):
                
                # Initialize the expected value for a specific action
                expected_val = 0.0
                
                # For every state we would land in...
                for sp_idx in range(S_joint):
                    
                    # ... obtain the probability that taking this action lands in sp
                    prob = P_joint.get((sp_idx, s_idx, a_idx), 0.0)
                    
                    # Only if the state is reachable...
                    if prob > 0:
                        
                        #  ...do we multiply the probability with V_{t+1} of being in that state
                        expected_val += prob * V[sp_idx, t+1]

                                        
                # Lastly, take the future value and multiply with the indicator Y
                action_values[a_idx] = Y_prod * expected_val
                
            # Pick action with the highest value and store in the policy
            pi_i_global[s_idx, t] = np.argmax(action_values)
            
            # Store the action index that maximizes value function
            best_a_idx = pi_i_global[s_idx, t]
            
            # Initialize a container for the expected value of taking the best action
            v_sum = 0.0
            
            # For each possible state you can land in given the best action...
            for sp_idx in range(S_joint):
                
                # ... we take the probability of landing in sp from the joint probability
                prob = P_joint.get((sp_idx, s_idx, best_a_idx), 0.0)
                
                # ... and multiply it by the value function at sp, adding it to v_sum to get our expected value
                v_sum += prob * V[sp_idx, t+1]
            
            # After finding the best action we assign it to the value matrix V
            V[s_idx, t] = Y_prod * v_sum
    
    
    time_end = time.time()
    
    print('Time to run the global policy:',np.round(time_end-time_start,2),'seconds.')
    
    #----------------------------------------------------------------------
    # End Function Here
    #----------------------------------------------------------------------
    
    # Calculate metrics for the current action_entropy
    pot_val, coll_lik, reach_red = global_metrics(V, pi_i_global, P_joint, Ps, x_0s, targ_raw_inds, T)
    print(pot_val, coll_lik, reach_red)
    
    # Store the results in a format matching your local feedback plot
    # We tile the values across 11 iterations (0-10) to create a horizontal baseline
    iterations = np.arange(0, 11)
    temp_df = pd.DataFrame({
        'BR_Iteration': np.tile(iterations, 2),
        'Value': [pot_val] * len(iterations) + [coll_lik] * len(iterations),
        'Metric': ['Potential'] * len(iterations) + ['Collision Likelihood'] * len(iterations),
        'Action Entropy': [f"{action_entropy:.1f}"] * 2 * len(iterations)
        
    })
    
    all_global_results.append(temp_df)
    print(f"Finished metrics for entropy: {action_entropy}")
    
    

# Combine all logged data
global_plot_data = pd.concat(all_global_results, ignore_index=True)

# Generate the final plot
sns.set_style("darkgrid")
g = sns.relplot(
    data=global_plot_data,
    x="BR_Iteration", 
    y="Value", 
    col="Metric", 
    hue="Action Entropy", 
    hue_order=[f"{0.1*(i+1):.1f}" for i in range(10)], # Ensures 0.1 to 1.0 order
    palette="viridis", 
    kind="line", 
    height=4, 
    aspect=0.8,
    facet_kws={'sharey': False}
)

# Format axes to match your Sarah Li baseline
for ax in g.axes.flat:
    ax.set_ylim(-0.05, 1.05)
    for line in ax.lines:
        line.set_linestyle('--') # Indicates theoretical limit

plt.show()


# Compute Occupancy: visualization/ heatmap over timestep that shows the highest probability of being that state at a given T. Think of expected cell to be in
# Test for when entropy is very high
# Verify: Global metrics are implemeneted correctly, Single or two player for two cases where you know the occupancy measure. Verify that the single player is going to the goal and the two player is avoiding each other
# Verify: Global metrics: If I gave two trajectories that are parallel vs trajs that are normal, what should they look like? Come up with test cases.
# Verify: How they perform to the local policy in the paper

# Comparison for Eq(32). numerator is the expected reacability, and the denomiator is the global policy which should be the upper bound/best case.
# Global should be smaller in denomiator (expected likeloihood of reaching target) but higher in the numerator
# Look into memory

# Line 141 should be using the new player rho
# For loop starting at line 113 should contain some condition for collisions, account for Y
# Compare them for a known setting, action, time, potential values
# For computation rnadomize over states and players