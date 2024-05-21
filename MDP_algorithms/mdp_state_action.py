# -*- coding: utf-8 -*-

import numpy as np
import util as ut
import random


"""
    Returns a rectangular MDP that is non-ergodic
    Grid with row = M, column = N, 
    p = main probability of going down a direction
    with_stay = if the agent can choose to stay in the current square
"""
def transitions(M, N, p, with_stay=False):
    A = 5 if with_stay else 4
    S = N*M
    P = np.zeros((S, S, A)) # destination state, origin state, action
    for i in range(M):
        for j in range(N):
            s = i*N + j
            left = i*N + j-1
            right = i*N + j + 1
            top = (i-1)*N + j
            bottom = (i+1)*N + j
            stay = i*N +j
            
            valid = []
            if s%N != 0:
                valid.append(left)
            if s%N != N-1:
                valid.append(right)
            if s >= N:
                valid.append(top)
            if s < (M*N - N):
                valid.append(bottom)
    
            lookup = {0: left, 1: right, 2: top, 3: bottom, 4:stay}
            for a in range(A):
                P[:,s,a] = ut.nonergodic_p(a, S, p, valid, lookup, s)   
    return P; 


def cost(S, A, T, target_s,  minimize=True):
    C = np.ones((S, A, T+1)) if minimize else np.zeros((S, A, T+1))
    # if minimize:
    #     C = np.ones((S, A, T+1))
    # else:
    #     C = np.zeros((S, A, T+1))
    # cost for agents in pick up mode
    C[target_s, :, :] = 0 if minimize else 1.
    return C


def pick_up_delivery_cost(Rows, Columns, A, T, pick_up_state, drop_offs, p_num, 
                         minimize=True):
    targ_rew = 0 if minimize else 1.
    S_sec = Rows*Columns
    S = S_sec * 2
    if minimize:
        C = np.ones((S, A, T+1))
    else:
        C = np.zeros((S, A, T+1))
    # cost for agents in pick up mode
    C[pick_up_state, :, :] = targ_rew
        
    # cost for agents delivery mode
    for drop_off in drop_offs:
        C[drop_off + S_sec, :, :] = targ_rew    
    return C

def set_up_cost(Rows, Columns, A, T, target_col, target_row,  p_num, 
                minimize=True, scal=1.):
    targ_rew = 0 if minimize else scal
    S = Rows * Columns
    if minimize:
        C = [np.ones((S, A, T+1)) for _ in range(p_num)]
    else:
        C = [np.zeros((S, A, T+1)) for _ in range(p_num)]
    for p in range(p_num):
        C[p][target_row*Columns + target_col[p], :, :] = targ_rew
    return C
    
def pol2dist(policy, x_0, P): 
    # policy is a 3D array for player
    # x is player p's initial state distribution at t = 0
    # returns player P's final distribution
    S, T = policy.shape
    x_arr = np.zeros((S, T+1))
    x_arr[:, 0] = x_0
    
    for t in range(T):
        # x_arr is the time state_density
        markov_chain = np.zeros((S,S))
        for cur_s in range(S):
            markov_chain[:, cur_s] = P[:, cur_s, int(policy[cur_s,t])]
        x_arr[:, t+1] = markov_chain.dot(x_arr[:, t]) 

    return x_arr

def occupancy(policy, P, initial_loc):
    # policy = ut.random_initial_policy_finite(Rows, Columns, A, T+1, p_num)
    S, _, _ = P.shape
    initial_x = np.zeros(S)
    initial_x[initial_loc] = 1.
    x = pol2dist(policy, initial_x, P)
    return x


def state_congestion_faster(Rows, Columns, modes, A, T, y):
    scal = 40.
    c_cost = np.zeros((modes*Rows*Columns, A, T+1))
    S = modes*Rows*Columns
    sum_actions = np.kron(np.eye(S), np.ones(A).T)
    sum_modes = np.kron(np.ones(modes).T, np.eye(Rows*Columns))
    expand_modes = np.kron(np.eye(Rows*Columns), np.ones(modes)).T
    # print(f'sum_modes shape {sum_modes.shape}')
    physical_dist = sum_modes.dot(sum_actions).dot(y)
    # print(f'physical dist shape is {physical_dist.shape}')
    congestion = scal*np.exp(scal*(physical_dist - 1))
    expanded_congestion = expand_modes.dot(congestion)
    # print(f'congestion_shape {congestion.shape}')
    for a in range(A):
        c_cost[:, a, :]  = expanded_congestion
    return c_cost

def state_congestion(Rows, Columns, modes, A, T, y):
    c_cost = np.zeros((modes*Rows*Columns, A, T+1))
    
    for x_ind in range(Columns):
        for y_ind in range(Rows):
            common_states = [y_ind * Columns + x_ind]
            for mode in range(modes-1):
                common_states.append(common_states[-1] + Rows*Columns)
            for t in range(T+1):
                density = sum([y[s*A:(s+1)*A, t] for s in common_states])
                congestion = 5* np.exp(5 * (density - 1))
                for s in common_states:
                    c_cost[s, :, t] += congestion
    return c_cost

def start_at_locs(pols, p_num, drop_offs, P, S, T):
    initial_x = [np.zeros(S) for _ in range(p_num)]
    x = [[] for _ in range(p_num)]
    for p in range(p_num):
        initial_x[p][drop_offs[p]] = 1.
        x[p].append(pol2dist(pols[:,:,:,p], initial_x[p], P[p], T))
    return x, initial_x




def execute_policy(initial_x, P, pols, T, targets):
    S, SA = P[0].shape 
    S_half = int(S/2)
    A = int(SA/S)
    # ind of first position the players are in
    trajs = [[np.where(x == 1)[0][0]] for x in initial_x] 
    # print(f' traj is {trajs}')
    
    _, _, T, p_num = pols.shape
    flat_pols = np.sum(pols, axis=0)
    collisions = {p: [] for p in range(p_num)}
    collision_timeline = [0 for t in range(T)]
    drop_off_counter = [[] for _ in range(p_num)]
    for t in range(T):
        for p_ind in range(p_num):
            cur_s = trajs[p_ind][-1]
            next_a = np.random.choice(
                np.arange(0,A),p=flat_pols[cur_s*A:(cur_s+1)*A, t, p_ind])
            cur_sa = cur_s*A+next_a
            # print(f'player {p_ind} current state{cur_s} action {next_a}')
            
            # sometimes these transition kernels don't sum to 1 
            # just normalize them as we go
            if sum(P[p_ind][:, cur_sa]) != 1:
                P[p_ind][:, cur_sa] = P[p_ind][:, cur_sa]/sum(P[p_ind][:, cur_sa])
            next_s = np.random.choice(np.arange(0,S), p=P[p_ind][:, cur_sa])
            # print(f' next state {next_s}')
            trajs[p_ind].append(1*next_s)
            # check pick up time
            if cur_s >= S_half and next_s < S_half:
                drop_off_counter[p_ind].append(1*t)
        # collision detection
        cur_pos = [trajs[p_ind][-1] for p_ind in range(p_num)]
        for p in range(p_num):  
            collisions[p].append(cur_pos.count(cur_pos[p])-1)
            collision_timeline[t] += cur_pos.count(cur_pos[p])-1
        # collisions.append(len(cur_pos) - len(set(cur_pos)))
        # if len(cur_pos) - len(set(cur_pos)) > 0:
        #     print(f'time {t} current positions {cur_pos}')
        # collisions += 
        drop_off_time = []
        for i in range(p_num):
            drop_offs = drop_off_counter[i]
            drop_off_time.append([
                drop_offs[j+1] - drop_offs[j] for j in range(len(drop_offs)-1)])
    return collisions, collision_timeline, drop_off_time
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
