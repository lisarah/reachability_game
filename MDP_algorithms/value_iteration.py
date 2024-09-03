# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations, permutations
import MDP_algorithms.mdp_state_action as mdp
import util as ut
from math import comb

def multiplicative_vi(pols, targs, inits, P, rhos, i, reachable_set):
    """ Compute the potential and the likelihood of no collision for two agents
    """
    print(f' running multiplicative value iteration')
    N = len(pols) 
    S, T = pols[0].shape
    A = 4
    # print(f' policy shape {policies[0].shape}')
    S_list = [s for s in range(S)]
    A_list = [a for a in range(A)]
    N_opp = [j for j in range(N)]; N_opp.remove(i)
    V = [{}, {}]
    pi = np.zeros((S,T))
    new_rho = None
    v_ind = 0
    all_reachable_si = {s: [] for s in S_list}
    for si in all_reachable_si.keys():
        for ai in A_list:
            all_reachable_si[si] += reachable_set[(si,ai)]
        all_reachable_si[si] = set(all_reachable_si[si])
    
    for s in permutations(S_list, N): # unique pairs of (s_1, ... s_N)
        X = np.array([ s_i == targs[i] for i, s_i in enumerate(s)])
        if X.all() == True:
            V[v_ind][s] = 1
    # print('current values')
    # print(V[v_ind])
    reachable_set = mdp.reachable_set(P[1]) 
    # print(reachable_set)
    tau = [{j: {} for j in N_opp} for _ in range(T)]# probability of player j's 2 state transition
    # computing trajectories
    for t in range(T-1,-1, -1):
        # collision free combinations of (s_1,...s_N)
        # print(f'time = {t}')
        v_ind = (v_ind + 1) % 2 
        v_prev = (v_ind + 1) % 2
        
        # first compute the conditional transitions of each player
        for j in N_opp:
            Pj = P[j]
            # print(f' opponent {j} at time {t}')
            pi_j = pols[j]
            rho_j = rhos[j]
            for sj in S_list:
                if rho_j[sj, t] > 1e-6:
                    # print(f'reachable at sa {(sj,int(pi_j[sj, t]))}=== {reachable_set[(sj,int(pi_j[sj, t]))]}')
                
                    # for hat_sj in reachable_set[(sj,int(pi_j[sj, t]))]:
                    #     print(f' at time {t}: transition {sj} to {hat_sj}')
                    at = int(pi_j[sj, t])
                    new_transitions = {(hat_sj,sj): Pj[hat_sj,sj,at]*rho_j[sj, t]
                        for hat_sj in reachable_set[(sj, at)]}
                    # print(f'------ new transitions at {t}, s {sj}')
                    # print({k: np.round(v,2) for k, v in new_transitions.items()})
                    tau[t][j].update(new_transitions)   
   
        test_state = 9
        test_time = T-1   
        # print(f'test time is {test_time}')
        opp_transition_list = [tau[t][j].keys() for j in N_opp]
        counter = 0
        # total = sum([len(all_reachable_si[si]) for si in S_list])
        for si in S_list:
            counter +=1
            eV = {hat_si: 0 for hat_si in all_reachable_si[si]}
            print(f' \r t = {t}  onto state {counter}/{S}      ', end='  ')
            # print(f'-------------------------------------------------')
            # print(f'\n reachable from s_i {all_reachable_si[si]}')
            # print(f'opponent list {opp_transition_list}')
            # print(f' on player {i}')
            opp_transition_list.insert(
                i, [(hat_si, si) for hat_si in all_reachable_si[si]])
            """This can be only iterated once """
            total_transitions = ut.cartesian_product(opp_transition_list)
            for hat_s_s in total_transitions:
                hat_s = tuple(hss[0] for hss in hat_s_s)
                Y = len(set(hat_s)) == len(hat_s) # no one experiences collision
                # print(f'-------------------------------------------------')
                # print(f' transition is {hat_s_s}')
                # print(f' next joint state is {hat_s}')
                # print(f' each transition is {[hat_s_s[j] for j in range(N)]}')
                # print(f' future s_i is {hat_s_s[i][0]}')
                if Y is True and hat_s in V[v_prev].keys():# V[t+1][hat_s] > 0
                    opp_density = np.prod([
                        tau[t][j][hat_s_s[j]] for j in N_opp])
                    if opp_density >= 1e-10:
                        eV[hat_s_s[i][0]] += opp_density*V[v_prev][hat_s]                  
            opp_transition_list.pop(i)    


            Q = [np.sum([
                P[i][hsi,si,a]*eV[hsi] for hsi in reachable_set[(si,a)]]) 
                for a in A_list]  
            # if pi[si, t] != np.argmax(Q) and t == 1:
            #     print(f' state {si} time {t} new policy is {np.argmax(Q)} Q is {np.max(Q)}')
            pi[si, t] = np.argmax(Q)
            # if t == test_time and si == test_state: 
            #     print(f' time {t} state {si} Q values {Q}')
            #     print(f' eV = {eV}')
            #     print(f' policy is {pi[si, t]}')
                
        
        counter = 0
        total = comb(S, N)
        for s in permutations(range(S), N): 
            counter +=1
            print(f' \r t = {t}  onto state {counter}/{total}      ', end='  ')
            # print(f' \r t = {t} \t\t  onto states {counter}/{S**N}    ', end='')
            # list future hat_sj that are reachable from (sj, aj)
            hats_list = [reachable_set[(sj,pj[sj, t])] 
                          for sj, pj in zip(list(s), pols)]
            # generate all combinations of hat_s
            reachable_hat_s = ut.cartesian_product(hats_list)
            # all nonzero P(hat_s | s)V(hat_s) must both
            # 1) come from reachable state
            # 2) V(hat_s) > 0
            # if t == 1 and s == (0,0): 
            #     print(f'previous keys = { V[v_prev].keys()}')
            #     print(f' s= 0 is in previous keys { 0 in V[v_prev].keys()}')
            valid_hat_s = V[v_prev].keys() & reachable_hat_s

            V_s = np.sum([V[v_prev][hat_s]*np.prod([Pj[hat_sj,sj,int(pij[sj,t])] 
                for Pj, hat_sj, sj, pij in zip(P, hat_s, s, pols)]) 
                for hat_s in valid_hat_s])

            # V_s = np.sum([np.prod([P_j[hat_s_j] for P_j, hat_s_j in zip(Ps, hat_s)])*V[v_prev][hat_s]
            #               for hat_s in V[v_prev].keys()])
            if V_s > (1e-1)**(t+3):
                V[v_ind][s] = V_s
            elif s in V[v_ind].keys():
                del V[v_ind][s]
      
    total_V = np.sum([V[v_ind][s]*np.prod([rho_i[si, 0] 
                                            for rho_i, si in zip(rhos, s)])
                      for s in V[v_ind].keys()])
    new_rho = mdp.occupancy(pi, P[i],  inits[i])
    # rhos = [mdp.occupancy(pols[p], P[p],  inits[p])  for p in range(2)]
    return total_V, pi, new_rho


def multiplicative_value(policies, targs, P, rhos, t):
    """ Compute the potential and the likelihood of no collision for two agents
    """
    N = len(policies)
    S, T = policies[0].shape
    T = T -1
    # print(f' policy shape {policies[0].shape}')
    S_list = [s for s in range(S)]
    V = [{}, {}]
    v_ind = 0
    for s in permutations(S_list, N): # unique pairs of (s_1, ... s_N)
        X = np.array([ s_i == targs[i] for i, s_i in enumerate(s)])
        if X.all() == True:
            V[v_ind][s] = 1
        
    for t in range(T-1):
        # collision free combinations of (s_1,...s_N)
        counter = 0
        v_ind = (v_ind + 1) % N 
        v_prev = (v_ind + 1) % N 
        for s in permutations(range(S), N): 
            print(f' \r t = {t} \t\t  onto states {counter}/{S**N}    ', end='')
            counter += 1
            # print(policies[0][s, t])
            # Ps = [P_i[:,s_i, int(pol[s_i, t])] for P_i, s_i, pol in zip(P, s, policies)]
            # if V_{t+1}(hat(s)) = 0, no point counting it
            # for s_i, rho_i in zip(s, rhos):
            #     if rho_i[s_i, t] <= 1e-10:
            #         break
                    
            V_s = np.sum([V[v_prev][hat_s]* np.prod([
                P_i[hat_s_j, s_i, int(pol[s_i, t])] 
                for P_i, hat_s_j, s_i, pol in zip(P, hat_s, s, policies)])
                for hat_s in V[v_prev].keys()])
            # V_s = np.sum([np.prod([P_j[hat_s_j] for P_j, hat_s_j in zip(Ps, hat_s)])*V[v_prev][hat_s]
            #               for hat_s in V[v_prev].keys()])
            if V_s > 0:
                V[v_ind][s] = V_s
            
    total_V = np.sum([V[v_ind][s]*np.prod([rho_i[si, 0] 
                                           for rho_i, si in zip(rhos, s)])
                      for s in V[v_ind].keys()])
    return total_V


def finite_reachability(P, T, targ_s, opponent_x=None): # always maximizing
    S, _, A = P.shape
    opp_x = opponent_x if opponent_x is not None else np.zeros((S,T+1))
    Vs = np.zeros((S, T+1)) # plural refers to time
    Vs[targ_s, T] = 1. 
    Vs = np.multiply(Vs, (1 - opp_x[targ_s, T]))
    pis = np.zeros((S, T)) # plural refers to time
    for t in range(T):
        t_ind =  T - 1 - t
        # Vs[:, t_ind+1] = np.multiply(Vs[:, t_ind+1], opp_x[:, t_ind+1])
        BO = np.einsum('ijk,i',P, Vs[:, t_ind+1])

        pis[:,t_ind] = np.argmax(BO, axis=1)
        Vs[:, t_ind] = 1. + np.multiply(np.max(BO, axis=1), 
                         np.ones((1,S)) - opp_x[:, t_ind])
    return Vs, pis
    
def value_iteration_tensor(P,c, T, minimize = True, g = 1.):
    plt.close('all')
    S, A, _ = c.shape
    optimize = np.min if minimize is True else np.max
    opt_arg = np.argmin if minimize is True else np.argmax
    pik = np.zeros((S, T+1))
    newpi = np.zeros((S,S*A, T+1))
    total_V = np.zeros((S,T+1))
    total_V[:, -1] =  optimize(c[:, :, T], axis=1)
    pik[:, -1] = opt_arg(c[:, :, T], axis=1)
    for s in range(S):
        newpi[s, int(s*A + pik[s,-1]), -1] = 1  
    for t in range(T):
        t_ind =  T - 1 - t
        BO = c[:,:,t_ind] + g*np.einsum('ijk,i',P,total_V[:, t_ind+1])
        pik[:,t_ind] = opt_arg(BO, axis=1)
        total_V[:, t_ind] =  optimize(BO, axis=1)
        for s in range(S):
            newpi[s, int(s*A + pik[s, t_ind]), t_ind] = 1    
    return total_V, newpi

def net_value_iteration(Transition_i, W, minimize = False):
    plt.close('all')
    optimize = np.min if minimize is True else np.max
    opt_arg = np.argmin if minimize is True else np.argmax
    # print(f' optimize is {optimize}')
    # print(f' opt_arg is {opt_arg}')
    # S, _, A= Pi.shape
    S, _,  T = W.shape
    _,_,A = Transition_i.shape
    # T = T_over - 1
    pi = np.zeros((S, T))
    V = np.zeros((S, T))
    # Vk[:, T] = optimize(BO[:,:,T_over], axis=1)
    for t in range(T):
        t_ind = T - t - 1 # T - 1 , T-2, T-3, ... 0
        # BO = np.einsum('ijk,ji->jk',Transition_i, W[:, :, t_ind])
        for s in range(S):
            
            BO = [Transition_i[:, s, a].dot(W[:,s,t_ind]) 
                  for a in range(A)]
            # print(f' time {t}, s,a table {BO.shape}')
            pi[s,t_ind] = opt_arg(BO)
            V[s, t_ind] =  optimize(BO)

    return V, pi

def value_iteration(P,c, T, minimize = False, g = 1.):
    # plt.close('all')
    optimize = np.min if minimize is True else np.max
    opt_arg = np.argmin if minimize is True else np.argmax
    # print(f' optimize is {optimize}')
    # print(f' opt_arg is {opt_arg}')
    S, _, A = P.shape
    # T = T_over - 1
    new_pi = np.zeros((S, T));
    # newpi = np.zeros((S,S*A, T));
    Vk = np.zeros((S, T+1))
    Vk[:, T] = c
    for t in range(T):
        t_ind = T - t - 1 # T - 1 , T-2, T-3, ... 0
        # print(f'P shape {P.shape} vk shape {Vk[:,t_ind+1].shape}')
        for s in range(S):
            BO = [P[:, s, a].dot(Vk[:,t_ind+1]) for a in range(A)]
            new_pi[s, t_ind] = opt_arg(BO)
            Vk[s, t_ind ] = optimize(BO)
    return Vk, new_pi

def value_iteration_dict(P,c, minimize = True, g = 1.):
    plt.close('all')
    optimize = np.min if minimize is True else np.max
    opt_arg = np.argmin if minimize is True else np.argmax
    S, A, T = c.shape
    # T = T_over - 1
    pik = np.zeros((S, T))
    newpi = np.zeros((S,S*A, T))
    Vk = np.zeros((S, T));
    BO = 1*c
    
    
    for t in range(T):
        t_ind = T - t - 1 # T - 1 , T-2, T-3, ... 0
        if t_ind  <  T - 1:
            # print(f'P shape {P.shape} vk shape {Vk[:,t_ind+1].shape}')
            for s in range(S):
                for a in range(A):
                    BO[s,a,t_ind] +=  g*sum([
                        prob[0]*Vk[prob[1], t_ind+1] for prob in P[(s, a)]])
        pik[:,t_ind] = opt_arg(BO[:,:,t_ind], axis=1)
        for s in range(S):
            Vk[s, t_ind] = BO[s, int(pik[s, t_ind]), t_ind]
            
    for s in range(S):
        for t in range(T):
            newpi[s, int(s*A + pik[s,t]), t] = 1.

    return Vk, newpi