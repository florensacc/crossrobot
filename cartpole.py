'''
Import packages
'''

import numpy as np
#import theano
#import theano.tensor as T
#theano.config.exception_verbosity='high'
import scipy as sp
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
import cPickle as pickle
import gzip
import os
import sys
import timeit

#################################################
# CARTPOLE SIMULATION AND TRAJECTORY GENERATION #
#################################################

def sim_cartpole(x0, u, dt=0.1, mc=10., mp=1.):
    '''
    Simulates cartpole starting at x0 with action u
    Modified from CS287 HW
    
    :type x0: np.array
    :param x0: starting point for the cartpole
    
    :type u: float or np.array
    :param u: action for the cartpole
    
    :type dt: float
    :param dt: time increment of simulation (default 0.1)
    
    :type mc: float
    :param mc: mass of the cart
    
    :type mp: float
    :param mp: mass of the pole
    '''
    
    def dynamics(x, u):
        l = 0.5
        g = 9.81
        T = 0.25
        s = np.sin(x[1])
        c = np.cos(x[1])
        
        xddot = (u + np.multiply(mp*s, l*np.power(x[3],2) + g*c))/(mc + mp*np.power(s,2))
        tddot = (-u*c - np.multiply(np.multiply(mp*l*np.power(x[3],2), c),s) - 
                 np.multiply((mc+mp)*g,s)) / (l * (mc + np.multiply(mp, np.power(s,2))))
        xdot = x[2:4]
        xdot = np.append(xdot, xddot)
        xdot = np.append(xdot, tddot)
        
        return xdot
    
    DT = 0.1
    t = 0
    while t < dt:
        current_dt = min(DT, dt-t)
        x0 = x0 + current_dt * dynamics(x0, u)
        t = t + current_dt
    
    return x0
    


def linearize_cartpole(x_ref, u_ref, dt, eps, mc=10., mp=1.):
    '''
    Linearizes the dynamics of cartpole around a reference point for use in an LQR controler.
    
    :type x_ref: np.array
    :param x_ref: reference point for linearization, i.e., the point to linearize around
    
    :type u_ref: np.array or float
    :param u_ref: reference action for initialization
    
    :type dt: float
    :type eps: float
    :type mc: float
    :type mp: float
    '''
    A = np.zeros([4,4])

    for i in range(4):
        increment = np.zeros([4,])
        increment[i] = eps
        A[:,i] = (sim_cartpole(x_ref + increment, u_ref, dt, mc, mp) - 
                  sim_cartpole(x_ref, u_ref, dt, mc, mp)) / (eps)
    
    B = (sim_cartpole(x_ref, u_ref + eps, dt, mc, mp) - sim_cartpole(x_ref, u_ref, dt, mc, mp)) / (eps)
    
    c = x_ref
    
    return A, B, c

def lqr_infinite_horizon(A, B, Q, R):
    '''
    Computes the LQR infinte horizon controller associated with linear dyamics A, B and quadratic cost Q, R
    '''
    nA = A.shape[0]

    if len(B.shape) == 1:
        nB = 1
    else:
        nB = B.shape[1]

    P_current = np.zeros([nA, nA])

    P_new = np.eye(nA)

    K_current = np.zeros([nB, nA])

    K_new= np.triu(np.tril(np.ones([nB,nA]),0),0)

    while np.linalg.norm(K_new - K_current, 2) > 1E-4:
        P_current = P_new
      
        K_current = K_new
        
        Quu = R + np.dot(np.dot( np.transpose(B), P_current), B)
        
        K_new = -np.linalg.inv(Quu) * np.dot(np.dot( np.transpose(B), P_current), A)
    
        P_new = Q + np.dot(np.dot( np.transpose(K_new), 
                                   R), 
                                   K_new) + np.dot(np.dot( np.transpose(A + np.dot(B.reshape(nA,1), K_new)),
                                                           P_current),
                                                           (A + np.dot(B.reshape(nA,1), K_new.reshape(1,nA)))
                          )
        
    return K_new, P_new, Quu
    
'''
Function to generate samples from the guidance trajectory
Updated to take cart mass and pole mass as parameters
'''
def gen_traj_guidance(x_init, x_ref, u_ref, K, variance, traj_size, dt, mc=10., mp=1.):
    xs = len(x_ref)
    
    if type(u_ref) == float:
        us = 1
    else:
        us = len(u_ref)
    
    x_traj = np.zeros([xs, traj_size])
    u_traj = np.zeros([us, traj_size])
    
    x_traj[:,0] = x_init
    u_traj[:,0] = np.random.multivariate_normal(np.dot(K, (x_traj[:,0] - x_ref) ) + u_ref, variance)
    
    for t in range(traj_size-1):
        x_traj[:,t+1] = sim_cartpole(x_traj[:,t], u_traj[:,t], dt, mc, mp)
        u_mean = np.dot(K, (x_traj[:,t] - x_ref) ) + u_ref
        u_traj[:,t+1] = np.random.multivariate_normal(u_mean, variance)
    
    return x_traj, u_traj

x_init_options = [
    np.array([0, np.pi - np.pi/4, 0, 0]),
    np.array([10, np.pi - np.pi/4, 0, 0]),
    np.array([-10, np.pi - np.pi/4, 0, 0]),
    np.array([0, np.pi + np.pi/4, 0, 0]),
    np.array([10, np.pi + np.pi/4, 0, 0]),
    np.array([-10, np.pi + np.pi/4, 0, 0]),
]

def sim_cartpole_ext(x0, u, dt):
    ''' 
    Simulates cartpole given an x0 provided that also encodes mc and mp in the last 2 entries
    '''
    mc = np.array(x0[-2])
    mp = np.array(x0[-1])
    xnew = np.array(x0)
    xnew[:4] = sim_cartpole(x0[:4], u, dt, mc, mp)
    return xnew

def linearize_cartpole_ext(x_ref, u_ref, dt, eps):
    ''' 
    Linearizes dynamics of cartpole where the x0 provided that also encodes mc and mp in the last 2 entries
    '''
    A = np.eye(6)
    B = np.zeros([6,])
    A[:4,:4], B[:4], c = linearize_cartpole(x_ref[:4], u_ref, dt, eps, x_ref[-2], x_ref[-1])
    c = x_ref
    return A, B, c

def gen_traj_guidance_ext(x_init, K, Quu, 
                          x_ref = np.array([0, np.pi, 0, 0]),
                          u_ref = 0.,
                          traj_size=500, dt = 0.01):
    '''
    Generate samples from the LQR policy
    '''
    
    #print x_init
    xs = len(x_init)
    
    if type(u_ref) == float:
        us = 1
    else:
        us = len(u_ref)
    
    #print x_init
    if len(x_ref) < xs:
        x_ref_ext = np.array(x_init)
        x_ref_ext[:4] = x_ref
    else:
        x_ref_ext = x_ref
    
    #print x_init
    x_traj = np.zeros([xs, traj_size])
    u_traj = np.zeros([us, traj_size])
    
    x_traj[:,0] = x_init
    u_traj[:,0] = np.random.multivariate_normal(np.dot(K, (x_traj[:,0] - x_ref_ext) ) + u_ref, Quu)
    
    #print x_init
    for t in range(traj_size-1):
        x_traj[:,t+1] = sim_cartpole_ext(x_traj[:,t], u_traj[:,t], dt)
        u_mean = np.dot(K, (x_traj[:,t] - x_ref_ext) ) + u_ref
        u_traj[:,t+1] = np.random.multivariate_normal(u_mean, Quu)
    
    return x_traj, u_traj
