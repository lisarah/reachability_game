# -*- coding: utf-8 -*-
from itertools import product, permutations
import numpy as np
import util as ut 


def potential(xs, targs):
    """ Compute the potential and the likelihood of no collision for two agents
    """
    N = len(xs)
    S,T = xs[0].shape
    T = T-1
    P_targ = 1.
    for p in range(N):
        P_targ = P_targ*xs[p][targs[p], T]  
    # print(f'p_targ is {P_targ}')
    P_no_collide = 1
    for t in range(T-1):
        P_collide = xs[0][:,t].dot(xs[1][:,t])
        P_no_collide = P_no_collide* P_no_collide*(1 - P_collide)
    return P_targ*P_no_collide, P_no_collide

def assign_next_V(t, s, valid_hat_s, prev_V, policies, P):
    V_s = np.sum([prev_V[hat_s]*np.prod([Pj[hat_sj,sj,int(pj[sj,t])] 
        for Pj, hat_sj, sj, pj in zip(P, hat_s, s, policies)]) 
        for hat_s in valid_hat_s])
    return V_s

def next_W(i, t, s, valid_hat_s, prev_V, policies, P, rhos, eps):
    # print(f' policy length {len(policies)} player {i}')
    N_opp = [j for j in range(len(policies)) if j != i]
    Slen, _, _ = P[0].shape
    Wt = np.zeros((Slen,Slen))
    # print(f' N_opp is {N_opp}')
    # W_s = np.sum([prev_V[hat_s]*np.prod([
    #     P[j][hat_s[j],s[j],int(policies[j][s[j],t])]*rhos[j][s[j], t] 
    #     for j in N_opp]) for hat_s in valid_hat_s])
    for hat_s in valid_hat_s:
        W_s_hat_s = prev_V[hat_s]*np.prod([
            P[j][hat_s[j],s[j],int(policies[j][s[j],t])]*rhos[j][s[j], t] 
            for j in N_opp])
        if W_s_hat_s >= eps:
            Wt[int(hat_s[i]), int(s[i])] += W_s_hat_s
    return Wt
    # print(f' current opponent policy {rhos[1][s[1], t] }')
    # print(f' current state is {s}')
    # print(f' current transition is {valid_hat_s}')
    # # print(f' current state adds {P[1][hat_s[1],s[j],int(policies[j][s[j],t])]*rhos[j][s[j], t]}')
    # # print(f' current policy {int(policies[1][s[1],t])}')
    # print(f'    W at state {s} adds {W_s}')
    # return W_s

def multiplicative_values(i, policies, targs, P, rhos, reachable_set):
    """ Compute the potential and the likelihood of no collision for two agents
    """
    print(' evaluating current potential')
    N = len(policies)
    # print(f' number of players is {N}')
    S, T = policies[0].shape
    T = T
    # print(f' policy shape {policies[0].shape}')
    S_list = [s for s in range(S)]
    V = [{}, {}]
    W = np.zeros((S, S, T))
    collision_V = [{}, {}]
    v_ind = 0
    for s in permutations(S_list, N): # unique pairs of (s_1, ... s_N)
        X = np.array([ s_i == targs[i] for i, s_i in enumerate(s)])
        if X.all() == True:
            V[v_ind][s] = 1
            # print(f'pair {s} achieves goals')
            
    for s in product(S_list, repeat=N):
        if any(s.count(elem)>1 for elem in list(s)):
            collision_V[v_ind][s] = 1
               
    # print({k: np.round(v, 2) for k, v in V[v_ind].items()})
    for t in range(T-1, -1, -1):
        # collision free combinations of (s_1,...s_N)
        eps = (1e-2)**(0.75*t+3)
        counter = 0
        v_ind = (v_ind + 1) % 2 
        v_prev = (v_ind + 1) % 2 
        # permutations don't count repeates (s_i never equal to s_j)
        for s in permutations(range(S), N):
            if counter % 1000 == 0:
                print(f' \r t = {t} \t\t  onto states {counter}/{S**N} '
                      f' value dict size {len(V[v_ind])}  ', end='')
            counter += 1
            prod_rho = np.prod([rhoj[sj, t] for rhoj, sj in zip(rhos, s)]) 
            if prod_rho <= eps:
                V_s = 0
                col_V_s = 0
            else:
                # list future hat_sj that are reachable from (sj, aj)
                hats_list = [reachable_set[(sj,pj[sj, t])] 
                             for sj, pj in zip(list(s), policies)]
                # if s == (20,20):
                #     print(hats_list)
                # generate all combinations of hat_s
                reachable_hat_s = ut.cartesian_product(hats_list)
                # to_print = ut.cartesian_product(hats_list)
                # all nonzero P(hat_s | s)V(hat_s) must both
                # 1) come from reachable state
                # 2) V(hat_s) > 0
                valid_hat_s = V[v_prev].keys() & reachable_hat_s
                valid_hat_s_col = collision_V[v_prev].keys() & \
                    ut.cartesian_product(hats_list)
                
                # print(f' from {s} reaches [', end = '  ')
                # for hats in V[v_prev].keys() & to_print:
                #     print(f'{hats},', end ='')
                # print(']')
                V_s = assign_next_V(t, s, valid_hat_s, V[v_prev], policies, P)
                col_V_s = assign_next_V(
                    t, s, valid_hat_s_col, collision_V[v_prev], policies, P)
                W[:, :, t] += next_W(i, t, s, valid_hat_s, V[v_prev], policies, 
                                     P, rhos, eps)

                # W[s[i], t] += np.prod([
                #     rhos[j][s[j], t] for rhoj, sj in zip(rhos, s)])*next_W(
                #     i, t, s, valid_hat_s, V[v_prev], policies, P, rhos)
                # V_s = np.sum([V[v_prev][hat_s]*np.prod([Pj[hat_sj,sj,int(pi[sj,t])] 
                #     for Pj, hat_sj, sj, pi in zip(P, hat_s, s, policies)]) 
                #     for hat_s in valid_hat_s])
            if V_s > eps:
                V[v_ind][s] = V_s
            elif s in V[v_ind].keys():
                del V[v_ind][s]    
            if col_V_s > eps:
                collision_V[v_ind][s] = col_V_s
            elif s in collision_V[v_ind].keys():
                del collision_V[v_ind][s]     
                # col_V_s = np.sum([V[v_prev][hat_s]*np.prod([Pj[hat_sj,sj,int(pi[sj,t])] 
                #     for Pj, hat_sj, sj, pi in zip(P, hat_s, s, policies)]) 
                #     for hat_s in valid_hat_s_col])

    print(f' \r final  value dict size {len(V[v_ind])}  ')    
    total_V = np.sum([V[v_ind][s]*np.prod([rho_i[si, 0] 
                                           for rho_i, si in zip(rhos, s)])
                      for s in V[v_ind].keys()])
    total_col_V = np.sum([collision_V[v_ind][s]*np.prod([rho_i[si, 0] 
                                           for rho_i, si in zip(rhos, s)])
                      for s in collision_V[v_ind].keys()])
    V[0].clear()
    V[1].clear()
    collision_V[0].clear()
    collision_V[1].clear()
    return total_V, total_col_V, W


def multiplicative_potential(policies, targs, P, rhos, reachable_set):
    """ Compute the potential and the likelihood of no collision for two agents
    """
    print(' evaluating current potential')
    N = len(policies)
    # print(f' number of players is {N}')
    S, T = policies[0].shape
    T = T
    # print(f' policy shape {policies[0].shape}')
    S_list = [s for s in range(S)]
    V = [{}, {}]
    collision_V = [{}, {}]
    v_ind = 0
    for s in permutations(S_list, N): # unique pairs of (s_1, ... s_N)
        X = np.array([ s_i == targs[i] for i, s_i in enumerate(s)])
        if X.all() == True:
            V[v_ind][s] = 1
            # print(f'pair {s} achieves goals')
            
    for s in product(S_list, repeat=N):
        if any(s.count(elem)>1 for elem in list(s)):
            collision_V[v_ind][s] = 1
        
    # print({k: np.round(v, 2) for k, v in V[v_ind].items()})
    for t in range(T-1, -1, -1):
        # collision free combinations of (s_1,...s_N)
        counter = 0
        v_ind = (v_ind + 1) % 2 
        v_prev = (v_ind + 1) % 2 
        # permutations don't count repeates (s_i never equal to s_j)
        for s in permutations(range(S), N):
            print(f' \r t = {t} \t\t  onto states {counter}/{S**N}    ', end='')
            counter += 1
            prod_rho = np.prod([rhoj[sj, t] for rhoj, sj in zip(rhos, s)]) 
            if prod_rho <=  1e-10:
                V_s = 0
                col_V_s = 0
            else:
                # list future hat_sj that are reachable from (sj, aj)
                hats_list = [reachable_set[(sj,pj[sj, t])] 
                             for sj, pj in zip(list(s), policies)]
                # if s == (20,20):
                #     print(hats_list)
                # generate all combinations of hat_s
                reachable_hat_s = ut.cartesian_product(hats_list)
                # to_print = ut.cartesian_product(hats_list)
                # all nonzero P(hat_s | s)V(hat_s) must both
                # 1) come from reachable state
                # 2) V(hat_s) > 0
                valid_hat_s = V[v_prev].keys() & reachable_hat_s
                valid_hat_s_col = collision_V[v_prev].keys() & \
                    ut.cartesian_product(hats_list)
                
                # print(f' from {s} reaches [', end = '  ')
                # for hats in V[v_prev].keys() & to_print:
                #     print(f'{hats},', end ='')
                # print(']')
                V_s = assign_next_V(t, s, valid_hat_s, V[v_prev], policies, P)
                col_V_s = assign_next_V(
                    t, s, valid_hat_s_col, collision_V[v_prev], policies, P)
                # V_s = np.sum([V[v_prev][hat_s]*np.prod([Pj[hat_sj,sj,int(pi[sj,t])] 
                #     for Pj, hat_sj, sj, pi in zip(P, hat_s, s, policies)]) 
                #     for hat_s in valid_hat_s])
            if V_s > (1e-1)**(t+3):
                V[v_ind][s] = V_s
            elif s in V[v_ind].keys():
                del V[v_ind][s]    
            if col_V_s > 0:
                collision_V[v_ind][s] = col_V_s
            elif s in collision_V[v_ind].keys():
                del collision_V[v_ind][s]     
                # col_V_s = np.sum([V[v_prev][hat_s]*np.prod([Pj[hat_sj,sj,int(pi[sj,t])] 
                #     for Pj, hat_sj, sj, pi in zip(P, hat_s, s, policies)]) 
                #     for hat_s in valid_hat_s_col])

        
    total_V = np.sum([V[v_ind][s]*np.prod([rho_i[si, 0] 
                                           for rho_i, si in zip(rhos, s)])
                      for s in V[v_ind].keys()])
    total_col_V = np.sum([collision_V[v_ind][s]*np.prod([rho_i[si, 0] 
                                           for rho_i, si in zip(rhos, s)])
                      for s in collision_V[v_ind].keys()])
    return total_V, total_col_V