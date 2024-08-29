#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:26:37 2024

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

Columns = 10
Rows = 5
T = 11
player_num =3
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

# ----- setting target state/initial state ------ #
# initialize the raw target and initial states of each player
targ_raw_inds = np.array([9, 49, 29])         
start_raw_inds = np.array([0, 40, 20])
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

# # ----- evaluating  the potential and setting up value functions  ------ #
# # current safety value: multiplicative_potential(policies, targs, S, P, rhos)
# # iterative best response
begin_t = time.time()
V, no_col_rate = game.multiplicative_potential(pols, targ_raw_inds, Ps, rhos, reachable_set)
end_t = time.time()
print(f' total time is {end_t - begin_t}')
col_rates = [no_col_rate]
potentials = [V]

# BR_iter = 10
# p = 0
# for ind in range(BR_iter):
#     print(f' ------- in best response iteration {ind}:'
#           f' potential = {np.round(potentials[-1], 5)}')
#     p = (p + 1) % player_num
    
#     Vk, new_pi, new_rho  = vi.multiplicative_vi(pols, targ_raw_inds, start_raw_inds, 
#                                   Ps, rhos, p, reachable_set)
#     pols[p] = new_pi
#     rhos[p] = new_rho
#     # print(f' policy difference {np.linalg.norm(pols[0] - ut.scrolling_policy_flat(S,  T))**2}')
#     V, no_col_rate = game.multiplicative_potential(pols, targ_raw_inds, Ps, rhos, reachable_set)
#     potentials.append(V)
#     col_rates.append(no_col_rate)
# print(potentials)                








    