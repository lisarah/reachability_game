#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:50:59 2024

@author: huilih
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
from itertools import product, combinations, permutations

Columns = 3
Rows = 3
T = 5
player_num = 2
"""
# test  two players with independent transitions 
    - both players start with optimal solution
    - player 1 should get to an individual optimal route
    - potential should be maximum 
"""

action_entropy = 0.95
# ----- defining MDP -----#
Ps = [mdp.transitions(Rows, Columns, p=action_entropy, with_stay=True) 
      for _ in range(player_num)]
reachable_set = mdp.reachable_set(Ps[0])

S, _, A = Ps[0].shape
pols = [ut.scrolling_policy_flat(S,  T) for _ in range(player_num)]

# # ----- setting target state/initial state ------ #
# # initialize the raw target and initial states of each player
start_raw_inds = np.array([0,6])
targ_raw_inds = np.array([2,8])     
# targs_initial_states = np.random.choice(range(Columns*Rows), 
#                              size=player_num*2, replace=False) 
# targ_raw_inds = targs_initial_states[:player_num]           
# start_raw_inds = targs_initial_states[player_num:]    

# turn targets into sinks (change transition)
for p in range(player_num):
    Ps[p][:, targ_raw_inds[p], 4] = np.zeros((S))
    Ps[p][targ_raw_inds[p], targ_raw_inds[p], 4] = 1.
    
rhos = [mdp.occupancy(pols[p], Ps[p],  start_raw_inds[p]) 
        for p in range(player_num)]

# # # ----- evaluating  the potential and setting up value functions  ------ #
# # # current safety value: multiplicative_potential(policies, targs, S, P, rhos)
# # # iterative best response
V, no_col = game.multiplicative_potential(
    pols, targ_raw_inds, Ps, rhos, reachable_set)
no_col_prob = [no_col]
potential = [V]

BR_iter = 1
p = 0
for ind in range(BR_iter):
    p = (p + 1) % player_num

    Vk, new_pi, new_rho  = vi.multiplicative_vi(
        pols, targ_raw_inds, start_raw_inds,   Ps, rhos, p, reachable_set)
    pols[p] = new_pi
    rhos[p] = new_rho
    new_pot, no_col_rate = game.multiplicative_potential(
        pols, targ_raw_inds, Ps, rhos, reachable_set)
    print(f' new potential is {new_pot}')
    potential.append(new_pot)
    no_col_prob.append(no_col_rate)
    # if potential[-1] == potential[-2]:
    #     break
print(potential)                








    