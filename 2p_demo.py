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

os_path = '/Users/huilih/anaconda3/bin:/Users/huilih/anaconda3/condabin:' \
          '/opt/homebrew/bin:/opt/homebrew/sbin:/Applications/Sublime Text.' \
          'app/Contents/SharedSupport/bin:/usr/local/bin:/System/Cryptexes/' \
          'App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/var/run/com.apple. ' \
          'security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/' \
          'com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/' \
              'run/com.apple.security.cryptexd/codex.system/bootstrap/usr/' \
                  'appleinternal/bin:/Library/TeX/texbin'
os.environ["PATH"] = os_path


""" Format matplotlib output to be latex compatible with gigantic fonts."""
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True) # change this back later, latex not found here.
mpl.rcParams.update({'font.size': 15})
mpl.rc('legend', fontsize='small')
# plt.rcParams.update(plt.rcParamsDefault)
# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'serif',
#     'font.size':15
# })

Columns = 10
Rows = 5
T = 20
player_num = 2

# set up trial
trial_potentials = []
trial_no_collisions = []
trial_data = pd.DataFrame({})

change_horizon = [(0.9, 5*i+1) for i in range(10)]
change_entropy = [(0.1*(i+1), 15) for i in range(10)]
for action_entropy, T in change_horizon:
    MCs = 100 # monte carlo trials
    
    # ----- defining MDP -----#
    Ps = [mdp.transitions(Rows, Columns, p=action_entropy, with_stay=True) 
          for p in range(player_num)]
    S, _, A = Ps[0].shape
    pols = ut.scrolling_policy(S, A, T+1, player_num)
    for tr in range(MCs):
        # ----- setting target state/initial state ------ #
        targs_x0s = np.random.choice(range(Columns*Rows), 
                                     size=player_num*2, replace=False) 
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
        trial_potentials.append(potential)
        trial_no_collisions.append(no_collisions)
        trial_data = pd.concat([trial_data, pd.DataFrame({
            'Trial': [tr]*len(potential), 
            'BR_Iteration': [i for i in range(len(potential))],
            'Potential': [p for p in potential],
            'Collision': [1 - no_col for no_col in no_collisions],
            'Horizon':[T]*len(potential),
            'Action Entropy':[action_entropy]*len(potential), 
            'P1 s_0': [x_0s[0]]*len(potential),
            'P2 s_0': [x_0s[1]]*len(potential),
            'P1 s_T': [x_0s[0]]*len(potential),
            'P2 s_T': [x_0s[1]]*len(potential)})], ignore_index=True)
        # trial_data = pd.concat([trial_data, pd.DataFrame({
        #     'Trial': [tr]*len(potential)*2, 
        #     'BR_Iteration': [i for i in range(len(potential))]*2,
        #     'Probability': [p for p in potential] + [1 - no_col for no_col in no_collisions],
        #     'Metric' : ['Potential']*len(potential) + ['Collision Likelihood']*len(potential),
        #     'Horizon':[T]*2*len(potential),
        #     'Action Entropy':[action_entropy]*len(potential)*2, 
        #     'P1 s_0': [x_0s[0]]*len(potential)*2,
        #     'P2 s_0': [x_0s[1]]*len(potential)*2,
        #     'P1 s_T': [x_0s[0]]*len(potential)*2,
        #     'P2 s_T': [x_0s[1]]*len(potential)*2})], ignore_index=True)
    
sns.set_style("darkgrid")
plot_1 = sns.relplot(kind="line",
    height = 5, aspect = 1.5, # col='N', 
    x="BR_Iteration", y='Collision', hue='Horizon', # style='type',   errorbar=("se", 5), 
    data=trial_data,
    # palette=sns.color_palette(),
);plt.show(block=False); # plt.yscale('log'); # plt.xscale('log');  #  
# plot_1 = sns.relplot(kind="line",
#     height = 5, aspect = 1.5, # col='N', 
#     x="BR_Iteration", y='Probability', hue='Metric', # style='type',   errorbar=("se", 5), 
#     data=trial_data,
#     # palette=sns.color_palette(),
# );  plt.show(block=False); # plt.xscale('log'); plt.yscale('log'); #  

# sns.set_style("darkgrid")
# plot_1 = sns.relplot(kind="line",
#     height = 5, aspect = 0.8, # col='N', 
#     x="BR_Iteration", y='value', errorbar=("se", 5), hue='variable', # style='type', 
#     data=pd.melt(trial_data, ['BR_Iteration', 'Action Entropy', 'Horizon T', 'Trial',
#                               'P1 s_0','P2 s_0','P1 s_T','P2 s_T']),
#     palette=sns.color_palette(),
# );  plt.xscale('log'); plt.yscale('log'); plt.show(block=False);

# # ------------- visualization --------------------- #
# total_player_costs = np.zeros(Columns * Rows)
# total_player_costs[targ_raw_inds] = 1.
# color_map, norm, _ = vs.color_map_gen(total_player_costs) 

# ax, value_grids, f = vs.init_grid_plot(Rows, Columns, total_player_costs)
# plt.show(block=False)

 
# # # print('visualizing now')
# vs.animate_traj(f'traj_ouput_{int(time.time())}.mp4', f, x_0s, pis, 
#                 total_player_costs, value_grids, Rows, Columns, Ps, Time=T)



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


    
    










    