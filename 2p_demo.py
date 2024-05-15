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
import random, time
import seaborn as sns
import matplotlib as mpl
import time


""" Format matplotlib output to be latex compatible with gigantic fonts."""
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=False) # change this back later, latex not found here.
mpl.rcParams.update({'font.size': 15})
mpl.rc('legend', fontsize='small')


Columns = 10
Rows = 5
T = 15
player_num = 2
# set up space 
targ_col = [Columns-1, 0] #, 0, 4, 0, 5, 1
targ_row = [Rows - 1, Rows - 1]
targ_raw_inds = [row*Columns + col for row, col in zip(targ_row, targ_col)]

# transition matrix : S x SA
Ps = [mdp.transitions(Rows, Columns, p=0.95, with_stay=True) 
      for p in range(player_num)]
S, _, A = Ps[0].shape

# cost matrix : S x A x (T+1)
Cs = [mdp.cost(S, A, T, targ_raw_inds[p],  minimize=False) 
      for p in range(player_num)]

# initial policies: x: list, each element: T
pols = ut.scrolling_policy(S, A, T+1, player_num)
initial_locs = [0, Columns-1]
# x, initial_x = mdp.occupancy_list(pols, Ps, T, player_num, initial_locs)


# reachability: 
Vs = [None for _ in range(player_num)]
pis = [None for _ in range(player_num)]
for p in range(player_num):
    V, pi = vi.finite_reachability(Ps[0], T, targ_raw_inds[p])
    Vs[p] = V
    pis[p] = pi    

# visualization
total_player_costs = np.zeros(Columns * Rows)
total_player_costs[targ_raw_inds] = 1.
color_map, norm, _ = vs.color_map_gen(total_player_costs) 

ax, value_grids, f = vs.init_grid_plot(Rows, Columns, total_player_costs)
plt.show(block=False)

 
# print('visualizing now')
vs.animate_traj(f'traj_ouput_{int(time.time())}.mp4', f, initial_locs, pis, 
                total_player_costs, value_grids, Rows, Columns, Ps, Time=T-1)



# # run frank-wolfe
# Iterations = 20 #100 # number of Frank wolf iterations
# V_hist= [[] for _ in range(player_num)]
# costs = [[] for _ in range(player_num)]

# gamma = 0.99
# steps = [1/(i+1) for i in range(Iterations)]
# begin_time = time.time()
# for i in range(Iterations):
#     print(f'\r on iteration {i}', end='   ')
#     next_distribution = []
#     y = sum([alpha[p] * x[p][-1] for p in range(player_num)])
#     for p in range(player_num):        
#         p_cost = C[p] - alpha[p] * st.state_congestion_faster(
#             Rows, Columns, mode_num, A, T, y) # 1.25 
#         costs[p].append(1*p_cost)
#         V, pol_new = dp.value_iteration_tensor(P_tensor[p], 1*p_cost, T, minimize=False)
#         V_hist[p].append(V)
#         pols[:,:,:,p] = (1-steps[i])*pols[:,:,:,p] + steps[i]*pol_new
#         x[p].append(st.pol2dist(pols[:,:,:,p],initial_x[p], P[p], T))     
# print(f'total time is {time.time() - begin_time}')   
    
# # # -------------  plot results  ---------------
# entries = 100
# res = {'Collisions': [], 't': [], 'player': []}
# wait_times = {'Average wait' : [], 'Max Wait': [],
#         'Player': [], 'Collisions': []}

# for ent in range(entries):
#     collisions, min_time = st.execute_policy(initial_x, P, pols, T, pick_ups)
#     # print(f'number of collisions is {collisions}')
#     # print(f'minimum time to target is {min_time}')
#     for p in range(player_num):
#         if len(min_time[p]) == 0:
#             average_wait = 0
#             max_wait = 0
#         else:
#             average_wait = sum(min_time[p])/len(min_time[p]) 
#             max_wait = max(min_time[p])
#         wait_times['Average wait'].append(average_wait)
#         wait_times['Max Wait'].append(max_wait)
#         wait_times['Player'].append(p)
#         wait_times['Collisions'].append(sum(collisions[p]))
#     # for t in range(T):
#     #     res['Collisions'].append(collisions[t])
#     #     res['t'].append(t)
# trials = pd.DataFrame.from_dict(res)

# sns.set_style("darkgrid")
# # columns = ['Collisions'] # , 'Time'
# # fig, axs = plt.subplots(figsize=(5,3), nrows=len(columns))
# # # axs = axs.flatten()
# # k = 0
# # for column in columns:
# #     # print(f'visualizing {column}')
# #     sns.lineplot(ax=axs, data=trials, x='t', y=column)
# #     axs.set(ylabel=column)
# #     k += 1
# # plt.show()

# columns = ['Average wait', 'Max Wait', 'Collisions'] # , 'Time'
# fig, axs = plt.subplots(figsize=(10,5), ncols=len(columns))
# axs = axs.flatten()
# k = 0
# for column in columns:
#     # print(f'visualizing {column}')
#     sns.lineplot(ax=axs[k], data=wait_times, x='Player', y=column)
#     axs[k].set(ylabel=column)
#     k += 1
# plt.show()

# V_hist_array = np.array(V_hist) # player, Iterations(alg), states, Timesteps+1
# # plot the value history as a function of states
# plt.figure()
# # plt.title('target pick up  state values')
# for p in range(player_num):
#     plt.plot(V_hist_array[p, 2:, pick_ups[p], 0], label=f'player {p}')
# plt.legend()
# plt.show()    

# plt.figure()
# plt.title('State values')
# for s in range(Rows*Columns):
#     plt.plot(V_hist_array[0,-1, s, :]) 
# plt.show()



# # cost_array = [np.array(costs[p]) for p in range(player_num)]
# # for p in range(player_num):
# #     plt.figure()
# #     plt.title(f'player {p} costs')
# #     for s in range(Rows*Columns):
# #         plt.plot(np.sum(cost_array[p][:, s, :, T-1], axis=1))
# #     plt.show()

# # # p1_costs = list(np.sum(cost_array[33, :,:,T-1],axis=1))
# # p1_costs = list(np.sum(cost_array[0][Iterations - 1, :,:,T],axis=1))
# # p1_values = V_hist_array[0,Iterations - 1, :, T-1]


    
    










    