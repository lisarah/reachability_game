# -*- coding: utf-8 -*-

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

