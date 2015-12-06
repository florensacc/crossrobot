#################################################
# CARTPOLE SIMULATION AND TRAJECTORY GENERATION #
#################################################
import numpy as np

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
    

def gen_traj_guidance(x_init, x_ref, u_ref, K, variance, traj_size, dt, mc=10., mp=1.):
    '''
    Function to generate samples from the guidance trajectory
    Updated to take cart mass and pole mass as parameters
    '''
    xs = len(x_ref)
    
    if (type(u_ref) == float) or (type(u_ref) == np.float32):
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
                          traj_size=500, dt = 0.1):
    '''
    Generate samples from the LQR policy
    '''
    
    xs = len(x_init)
    
    if (type(u_ref) == float) or (type(u_ref) == np.float32):
        us = 1
    else:
        us = len(u_ref)
    
    if len(x_ref) < xs:
        x_ref_ext = np.array(x_init)
        x_ref_ext[:4] = x_ref
    else:
        x_ref_ext = x_ref
    
    x_traj = np.zeros([xs, traj_size])
    u_traj = np.zeros([us, traj_size])
    
    x_traj[:,0] = x_init
    u_traj[:,0] = np.random.multivariate_normal(np.dot(K, (x_traj[:,0] - x_ref_ext) ) + u_ref, Quu)
    
    for t in range(traj_size-1):
        x_traj[:,t+1] = sim_cartpole_ext(x_traj[:,t], u_traj[:,t], dt)
        u_mean = np.dot(K, (x_traj[:,t] - x_ref_ext) ) + u_ref
        u_traj[:,t+1] = np.random.multivariate_normal(u_mean, Quu)
    
    return x_traj, u_traj

def gen_train_data(LQR_start, LQR_controller, LQR_var, num_traj=10, traj_size=500, pred_mass = False, dt=0.1):
    x_traj_list = []
    u_traj_list = []
    # Generate num_traj sample trajectories from our LQR policy for each of training, validation, and test
    if type(LQR_start) == list:
        n_guidance = len(LQR_start)
        for i in range(3): # training, validation, test
            for j in range(n_guidance): # starting positions
                for k in range(num_traj): # generate this many trajectories
                    if len(LQR_start) != len(LQR_controller):
                        raise TypeError(
                            'Please provide a LQR controller and a variance for each x_init'
                        )

                    x_traj1, u_traj1 = gen_traj_guidance_ext(LQR_start[j], LQR_controller[j], LQR_var[j],
                                                            traj_size=traj_size, dt=dt)
                    x_traj_list.append(x_traj1)
                    u_traj_list.append(u_traj1)

    else:
        n_guidance = 1
        for t in range(3*num_traj):
            x_traj1, u_traj1 = gen_traj_guidance_ext(LQR_start, LQR_controller, LQR_var, traj_size=traj_size, dt=dt)
            x_traj_list.append(x_traj1)
            u_traj_list.append(u_traj1)
    
    if pred_mass:
        temp_u = [path[4:,:-1] for path in x_traj_list]
        x_traj_list = [ np.vstack( 
                                    (path[:4,:-1],
                                     upath[:,:-1],
                                     path[:4, 1:]
                                     )
                                  )
                       for path, upath in zip(x_traj_list, u_traj_list)
                       ]
        
        u_traj_list = temp_u
        
    
    ts_x = np.concatenate(x_traj_list[:n_guidance*num_traj], axis=1).T
    ts_u = np.concatenate(u_traj_list[:n_guidance*num_traj], axis=1).T
    vs_x = np.concatenate(x_traj_list[n_guidance*num_traj:2*n_guidance*num_traj], axis=1).T
    vs_u = np.concatenate(u_traj_list[n_guidance*num_traj:2*n_guidance*num_traj], axis=1).T
    te_x = np.concatenate(x_traj_list[2*n_guidance*num_traj:3*n_guidance*num_traj], axis=1).T
    te_u = np.concatenate(u_traj_list[n_guidance*num_traj:2*n_guidance*num_traj], axis=1).T
    
    return ts_x, ts_u, vs_x, vs_u, te_x, te_u
