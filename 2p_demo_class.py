# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 17:45:15 2022

@author: Sarah Li
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
    
def local_feedback_policy(Rows,Columns,T,player_num,targs_x0s,action_entropy):
    
    trial_potentials = []
    trial_no_collisions = []
    trial_data = pd.DataFrame({})
    
    MCs = 100 # monte carlo trials
    
    # ----- defining MDP -----#
    Ps = [mdp.transitions(Rows, Columns, p=action_entropy, with_stay=True) 
          for p in range(player_num)]
    S, _, A = Ps[0].shape
    pols = ut.scrolling_policy(S, A, T+1, player_num)
    time_start = time.time()
    for tr in range(MCs):
        # ----- setting target state/initial state ------ #
        #targs_x0s = np.random.choice(range(Columns*Rows), 
                                     #size=player_num*2, replace=False) 
        targ_raw_inds = targs_x0s[:player_num]           
        x_0s = targs_x0s[player_num:]
        
        # turn targets into sinks
        for p in range(player_num):
            Ps[p][:, targ_raw_inds[p], 4] = np.zeros((S))
            Ps[p][targ_raw_inds[p], targ_raw_inds[p], 4] = 1.
            
        # iterative best response
        Vs = [None for _ in range(player_num)]
        pols = ut.scrolling_policy_flat(S,  T+1)
        pis = [pols for _ in range(player_num)]
        xs = [mdp.occupancy(pis[p], Ps[p],  x_0s[p]) for p in range(player_num)]
        potential = []
        no_collisions = []
        pot, no_col = game.potential(xs, targ_raw_inds)
        potential.append(pot)
        no_collisions.append(no_col)
        
        BR_iter = 5
        for ind in range(BR_iter):
            for p in range(player_num):
                opponent = (p+1)%player_num
                V, pi = vi.finite_reachability(Ps[0], T, targ_raw_inds[p], xs[opponent]) # 
                Vs[p] = V
                pis[p] = pi  
                xs[p] = mdp.occupancy(pis[p], Ps[p],  x_0s[p]) 
                if isinstance(xs[0], np.ndarray) and isinstance(xs[1], np.ndarray):
                    pot, no_col = game.potential(xs, targ_raw_inds)
                    potential.append(pot)
                    no_collisions.append(no_col)
        time_end = time.time()
        time_elapsed = time_end-time_start
        trial_potentials.append(potential)
        trial_no_collisions.append(no_collisions)

        trial_data = pd.concat([trial_data, pd.DataFrame({
            'Trial': [tr]*len(potential)*2, 
            'BR_Iteration': [i for i in range(len(potential))]*2,
            'Value': [p for p in potential] + [1 - no_col for no_col in no_collisions],
            'Metric' : ['Potential']*len(potential) + ['Collision Likelihood']*len(potential),
            'Horizon':[T]*2*len(potential),
            #'Action Entropy':[action_entropy]*len(potential)*2, 
            'Action Entropy': [str(round(action_entropy, 2))] * len(potential) * 2,
            'P1 s_0': [x_0s[0]]*len(potential)*2,
            'P2 s_0': [x_0s[1]]*len(potential)*2,
            'P1 s_T': [x_0s[0]]*len(potential)*2,
            'P2 s_T': [x_0s[1]]*len(potential)*2})], ignore_index=True)
        
    return trial_data,time_elapsed




    
    










    