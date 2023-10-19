#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as ca_tools
import matplotlib.pyplot as plt

import numpy as np
import time
from draw import Draw_MPC_tracking
from draw import Draw_MPC_Obstacle


def shift_movement(T, t0, x0, u, x_, f):
    # f_value = f(x0, u[:, 0])
    # st = x0 + T*f_value
    k1 = f(x0, u[:, 0])
    k2 = f(x0 + T/2.0*k1, u[:, 0])
    k3 = f(x0 + T/2.0*k2, u[:, 0])
    k4 = f(x0 + T*k3, u[:, 0])

    st = x0 + (T/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    t = t0 + T
    u_end = ca.horzcat(u[:, 1:], u[:, -1])
    x_n = ca.horzcat(x_[:, 1:], x_[:, -1])
    return t, st, u_end, x_n

#def desired_command_and_trajectory(t, T, x0_, N_):
    # initial state / last state
    x_ = x0_.reshape(1, -1).tolist()[0]
    u_ = []
    # states for the next N_ trajectories
    for i in range(N_):
        t_predict = t + T*i
        x_ref_ = 0.5 * t_predict
        y_ref_ = 1.0
        theta_ref_ = 0.0
        v_ref_ = 0.5
        omega_ref_ = 0.0
        if x_ref_ >= 12.0:
            x_ref_ = 12.0
            v_ref_ = 0.0
        x_.append(x_ref_)
        x_.append(y_ref_)
        x_.append(theta_ref_)
        u_.append(v_ref_)
        u_.append(omega_ref_)
    # return pose and command
    x_ = np.array(x_).reshape(N_+1, -1)
    u_ = np.array(u_).reshape(N, -1)
    return x_, u_
#def desired_command_and_trajectory(t, T, x0_, N_):
    r = 2.0 # radius of the circle
    w = 0.1 # angular velocity to move on the circle (adjust to desired speed)
    
    # initial state / last state
    x_ = x0_.reshape(1, -1).tolist()[0]
    u_ = []
    # states for the next N_ trajectories
    for i in range(N_):
        t_predict = t + T*i
        x_ref_ = r * np.cos(w * t_predict)
        y_ref_ = r * np.sin(w * t_predict)
        theta_ref_ = w * t_predict
        v_ref_ = w * r
        omega_ref_ = w
        x_.append(x_ref_)
        x_.append(y_ref_)
        x_.append(theta_ref_)
        u_.append(v_ref_)
        u_.append(omega_ref_)

    # return pose and command
    x_ = np.array(x_).reshape(N_+1, -1)
    u_ = np.array(u_).reshape(N, -1)
    return x_, u_
import numpy as np

def desired_command_and_trajectory(t, T, x0_, N_):
    # initial state / last state
    x_ = x0_.reshape(1, -1).tolist()[0]
    u_ = []

    # Define circle parameters
    r = 1.0  # radius of the circle
    angular_velocity = 0.5  # how fast we move around the circle

    # states for the next N_ trajectories
    for i in range(N_):
        t_predict = t + T * i
        theta_ref_ = angular_velocity * t_predict  # angle at time t_predict
        x_ref_ = r * np.cos(theta_ref_)
        y_ref_ = r * np.sin(theta_ref_)

        v_ref_ = r * angular_velocity  # tangential velocity around the circle
        omega_ref_ = angular_velocity

        x_.extend([x_ref_, y_ref_, theta_ref_])
        u_.extend([v_ref_, omega_ref_])

    # return pose and command
    x_ = np.array(x_).reshape(N_+1, -1)
    u_ = np.array(u_).reshape(N, -1)
    return x_, u_

def get_estimated_result(data, N_):
    x_ = np.zeros((N_+1, 3))
    u_ = np.zeros((N_, 2))
    for i in range(N_):
        u_[i] = data[i*5:i*5+2].T
        x_[i] = data[i*5+2:i*5+5].T
    x_[-1] = data[-3:].T 
    return  u_, x_



if __name__ == '__main__':
    T = 0.1 # sampling time [s]
    N = 20 # prediction horizon
    rob_diam = 0.3 # [m]
    v_max = 0.6
    omega_max = np.pi/2.0

    states = ca_tools.struct_symSX([
        (
            ca_tools.entry('x'),
            ca_tools.entry('y'),
            ca_tools.entry('theta')
        )
    ])
    x, y, theta = states[...]
    n_states = states.size

    controls  = ca_tools.struct_symSX([
        (
            ca_tools.entry('v'),
            ca_tools.entry('omega')
        )
    ])
    v, omega = controls[...]
    n_controls = controls.size

    ## rhs
    rhs = ca_tools.struct_SX(states)
    rhs['x'] = v*ca.cos(theta)
    rhs['y'] = v*ca.sin(theta)
    rhs['theta'] = omega

    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    optimizing_target = ca_tools.struct_symSX([
        (
            ca_tools.entry('U', repeat=N, struct=controls),
            ca_tools.entry('X', repeat=N+1, struct=states)
        )
    ])
    U, X, = optimizing_target[...] # data are stored in list [], notice that ',' cannot be missed

    ### basically here are the parameters that for trajectory definition
    current_parameters = ca_tools.struct_symSX([
        (
            ca_tools.entry('U_ref', repeat=N, struct=controls),
            ca_tools.entry('X_ref', repeat=N+1, struct=states)
        )
    ])
    U_ref, X_ref,  = current_parameters[...]

    ### define
    Q = np.array([[1.0, 0.0, 0.0],[0.0, 5.0, 0.0],[0.0, 0.0, .1]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    #### cost function
    obj = 0 #### cost
    #### constrains
    g = [] # equal constrains
    g.append(X[0]-X_ref[0]) # initial condition constraints

    #for i in range(N):
     #   state_error_ = X[i] - X_ref[i+1]
      #  control_error_ = U[i] - U_ref[i]
       # obj = obj + ca.mtimes([state_error_.T, Q, state_error_]) + ca.mtimes([control_error_.T, R, control_error_])
        #x_next_ = f(X[i], U[i])*T + X[i]
        #g.append(X[i+1] - x_next_)
    obs_x = 1.0
    obs_y = 0.0
    obs_diam = 0.5
    for i in range(N):
        state_error_ = X[i] - X_ref[i+1]
        control_error_ = U[i] - U_ref[i]
        obj = obj + ca.mtimes([state_error_.T, Q, state_error_]) + ca.mtimes([control_error_.T, R, control_error_])
        k1 = f(X[i], U[i])
        k2 = f(X[i] + T/2.0*k1, U[i])
        k3 = f(X[i] + T/2.0*k2, U[i])
        k4 = f(X[i] + T*k3, U[i])

        x_next_ = X[i] + (T/6.0)*(k1 + 2*k2 + 2*k3 + k4)
       
        g.append(X[i+1] - x_next_)
        #g.append((X[i][0] - 1)**2 + (X[i][1] - 0)**2 - 0.12**2)  #-(rob_diam/2.+obs_diam/2.)
        
        g.append(((X[i][0] - obs_x)**2 + (X[i][1] - obs_y)**2 )-(rob_diam/2.+obs_diam/2.)**2)
        #g.append(-((X[i][0] - obs_x)**2 + (X[i][1] - obs_y)**2 )+(rob_diam+obs_diam))

    
    nlp_prob = {'f': obj, 'x': optimizing_target, 'p':current_parameters, 'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter':2000, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
    # lbg = 0
    # ubg = 0
    lbg = [0.0, 0.0, 0.0]
    ubg = [0.0, 0.0, 0.0]
    lbx = []
    ubx = []
    ## add constraints to control and statesn notice that for the N+1 th state
    for _ in range(N):
        # For the Dynamic Constraint
        lbg.extend([0.0, 0.0, 0.0])
        ubg.extend([0.0, 0.0, 0.0])
        # For Obstacle Constraint
        ubg.append(np.inf)
        lbg.append(0.0)
        # States constraints
        lbx.append(-v_max)
        lbx.append(-omega_max)
        lbx.append(-np.inf)
        lbx.append(-np.inf)
        lbx.append(-np.inf)

        ubx.append(v_max)
        ubx.append(omega_max)
        ubx.append(np.inf)
        ubx.append(np.inf)
        ubx.append(np.inf)
    # for the N+1 state
    lbx.append(-np.inf)
    lbx.append(-np.inf)
    lbx.append(-np.inf)
    ubx.append(np.inf)
    ubx.append(np.inf)
    ubx.append(np.inf)

    # Simulation
    t0 = 0.0
    init_state = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)# initial state
    current_state = init_state.copy()
    u0 = np.array([0.0, 0.0]*N).reshape(-1, 2).T# np.ones((N, 2)) # controls
    next_trajectories = np.tile(current_state.reshape(1, -1), N+1).reshape(N+1, -1)
    next_states = next_trajectories.copy()
    next_controls = np.zeros((N, 2))
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [t0] # for the time
    xx = []
    sim_time = 30.0
    ## start MPC
    mpciter = 0 # mpc iteration time
    start_time = time.time()
    index_t = []
    ### inital test
    c_p = current_parameters(0) # references
    init_input = optimizing_target(0)
    while(mpciter-sim_time/T<0.0):
        current_time = mpciter * T # current time (norm (x0-xs),2) > 1e-2 && mpciter-sim_time/T<0.0)
        ## obtain the desired trajectory, note that, the input should be (N*, states*), then the output will turn to (states*, N*)
        c_p['X_ref', lambda x:ca.horzcat(*x)] = next_trajectories.T
        c_p['U_ref', lambda x:ca.horzcat(*x)] = next_controls.T
        ## set parameter
        init_input['X', lambda x:ca.horzcat(*x)] = next_states.T
        init_input['U', lambda x:ca.horzcat(*x)] = u0
        t_ = time.time()
        res = solver(x0=init_input, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time()- t_)
        estimated_opt = res['x'].full() # the feedback is in the series [u0, x0, u1, x1, ...]
        u_res, x_m = get_estimated_result(estimated_opt, N) # the result are in form (N*, states)
        x_c.append(x_m)
        u_c.append(u_res[0])
        t_c.append(t0)
        t0, current_state, u0, next_states = shift_movement(T, t0, current_state, u_res.T, x_m, f)
        current_state = ca.reshape(current_state, -1, 1)
        current_state = current_state.full()
        xx.append(current_state)
        next_trajectories, next_controls = desired_command_and_trajectory(t0, T, current_state, N)
        mpciter = mpciter + 1
        
    print(mpciter)
    t_v = np.array(index_t)
    print(t_v.mean()) 
    print((time.time() - start_time)/(mpciter))
    draw_result = Draw_MPC_tracking(rob_diam=rob_diam, init_state=init_state, robot_states=xx )
    draw_result = Draw_MPC_Obstacle(rob_diam=0.3, init_state=init_state, robot_states=xx, obstacle=np.array([obs_x, obs_y, obs_diam/2.]), export_fig=False)

    # draw_result.draw_trajectory()



#Draw_MPC_tracking(rob_diam=rob_diam, init_state=init_state, robot_states=xx)
