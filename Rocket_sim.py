# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:13:43 2026

@author: lucas
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(69)

params = {
    "propellant mass":   0.183,                          # kg                   fuel mass
    "dry mass":          0.5,                            # kg                   (structure + payload)
    "burn_time":         3,                              # s                    how long the motor fires (overwritten by thrust curve)
    "g":                 9.81,                           # m/s^2                gravitational acceleration
    "rho0":              1.225,                          # kg/m^3               sea - level air density
    "H":                 8500,                           # m                    scale height - height which density drops by factor of e
    "dt":                0.1,                            # s                    timestep
    "Cd":                0.5,                            #                      drag coefficient for a rough rocket (approx)
    "A":                 np.pi * 0.020**2,               # m^2                  frontal area (40mm airframe,  r = 0.02)
    "theta_deg":         90,                             # degrees              angle of rocket launch
}

# Thrust curve file cleaner

def load_thrust_curve(filename):
    times = [0.0]
    thrusts = [0.0]
    header_seen = False
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith(";"):
                continue
            if not header_seen:
                header_seen = True
                continue
            parts = line.split()
            times.append(float(parts[0]))
            thrusts.append(float(parts[1]))
    return times, thrusts

# Loads the thurst-curve file

times, thrusts = load_thrust_curve("AeroTech_HP-I140W.eng")
params["curve_times"] = times
params["curve_thrusts"] = thrusts
params["burn_time"] = times[-1]
    
# Force components 

def thrust(t, p):
    return np.interp(t, p["curve_times"], p["curve_thrusts"])

def rho(y, p):   # air density formula (used to calculate drag)
    rho_y = p["rho0"] * np.exp(-y / p["H"])
    return rho_y
   
def drag(y, vx, vy, p):
    rho_y = rho(y, p)
    speed = np.sqrt(vx**2 + vy**2)
    F_drag = -0.5 * rho_y * p["Cd"] * p["A"] * speed * np.array([vx, vy])
    return F_drag

def mass_flow(t, p):
    burn_rate = p["propellant mass"] / p["burn_time"]
    if t < p["burn_time"]:
        return burn_rate
    else:
        return 0.0
    
def net_force(state, t, p):
    x, y, vx, vy, m = state
    theta = np.radians(p["theta_deg"])
    thrust_x = thrust(t, p) * np.cos(theta)
    thrust_y = thrust(t, p) * np.sin(theta)
    F_drag = drag(y, vx, vy, p)
    Fx = thrust_x + F_drag[0]
    Fy = thrust_y - (m * p["g"]) + F_drag[1]
    return np.array([Fx, Fy])

# Integrator - single step Euler method

def step_euler(state, t, p):
    x, y, vx, vy, m = state
    dt = p["dt"]
    a = net_force(state, t, p) / m
    ax, ay = a
    new_vx = vx + ax * dt
    new_vy = vy + (ay * dt)
    new_x = x + new_vx * dt
    new_y = y + (new_vy * dt)
    new_m = m - mass_flow(t, p) * dt # v2: mass of fule lost as rocket burns
    return np.array([new_x, new_y, new_vx, new_vy, new_m])

# Creating the derivatives

def derivatives(t, state, p):
    x, y, vx, vy, m = state
    F = net_force(state, t, p)
    dydt = vy
    dxdt = vx
    dvydt = F[1] / m 
    dvxdt = F[0] / m
    dmdt = -mass_flow(t, p)
    return np.array([dxdt, dydt, dvxdt, dvydt, dmdt])

# RK4 Integrator

def step_RK4(state, t, p):                  
    dt = p["dt"]
    k1 = derivatives(t, state, p)
    k2 = derivatives(t + dt/2, state + dt/2 * k1, p)
    k3 = derivatives(t + dt/2, state + dt/2 * k2, p)
    k4 = derivatives(t + dt, state + dt * k3, p)
    new_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state
    
# Simulation loop

def simulate(p, step_fn):
    m0 = p["dry mass"] + p["propellant mass"]
    state = np.array([0.0, 0.0, 0.0, 0.0, m0]) # Initial state on ground at rest and full mass
    t = 0.0
    times, altitudes, xs, velocities = [], [], [], []
    burnout_t, apogee_t = None, None
    launched = False
    
    
    while True:
        # record the current state before stepping
        times.append(t)
        xs.append(state[0])
        altitudes.append(state[1])
        velocities.append(state[3])
        
        # detect burnout: the first moment the motor stops firing
        if burnout_t is None and t >= p["burn_time"]:
            burnout_t = t

        # advance one step
        prev_vy = state[3]
        state = step_fn(state, t, p)
        t += p["dt"]

        # detect apogee: velocity flips from climbing (+) to falling (-)
        if apogee_t is None and prev_vy > 0 and state[3] <= 0:
            apogee_t = t

        # mark that rocket left the pad
        if state[1] > 0:
            launched = True

        # stop once rocket launched wont come back to ground
        if launched and state[1] <= 0:
            break

        if t > 60.0:   # safety cap
            break

    return times, altitudes, xs, velocities, burnout_t, apogee_t

# Sensitivity analysis 

def angle_ranges(p, angles):
    apogees = []
    ranges = []
    for ang in angles:
        p["theta_deg"] = ang
        times, altitudes, xs, velocities, burnout_t, apogee_t = simulate(p, step_RK4)
        apogees.append(max(altitudes))
        ranges.append(xs[-1])
    return apogees, ranges

# Monte Carlo layer

def monte_carlo (p, N):
    apogees = []
    ranges = []
    
    Cd_nom =  p["Cd"]
    mass_nom = p["dry mass"]
   
    for i in range(N):
        p["Cd"] = np.random.normal(Cd_nom, 0.10*Cd_nom)
        p["dry mass"] = np.random.normal(mass_nom, 0.03*mass_nom)
        times, altitudes, xs, velocities, burnout_t, apogee_t = simulate(p, step_RK4)
        apogees.append(max(altitudes))
        ranges.append(xs[-1])
    print("apogee mean:", round(np.mean(apogees), 1))
    print("apogee std: ", round(np.std(apogees), 1))
    print("range mean: ", round(np.mean(ranges), 1))
    print("range std:  ", round(np.std(ranges), 1))
    print("apogee 5-95:", np.round(np.percentile(apogees, [5, 95]), 1))
    print("range 5-95: ", np.round(np.percentile(ranges, [5, 95]), 1))
    return apogees, ranges

# Graph output

def plot_comparison(euler, rk4):
    t_e, alt_e, xs_e, vel_e, burnout_t, apogee_t = euler
    t_r, alt_r, xs_r, vel_r, _, _ = rk4

    plt.figure()                                         # altitude - time plot
    plt.plot(t_e, alt_e, label='Euler')
    plt.plot(t_r, alt_r, label='RK4')
    if burnout_t is not None:
        plt.axvline(burnout_t, color='orange', linestyle='--', label='burnout')
    if apogee_t is not None:
        plt.axvline(apogee_t, color='red', linestyle='--', label='apogee')
    plt.xlabel('time (s)'); plt.ylabel('altitude (m)')
    plt.title('Altitude vs Time — Euler vs RK4'); plt.legend()

    plt.figure()                                         # velocity - time plot
    plt.plot(t_e, vel_e, label='Euler')
    plt.plot(t_r, vel_r, label='RK4')
    plt.axhline(0, color='grey', linewidth=0.8)
    if burnout_t is not None:
        plt.axvline(burnout_t, color='orange', linestyle='--', label='burnout')
    if apogee_t is not None:
        plt.axvline(apogee_t, color='red', linestyle='--', label='apogee')
    plt.xlabel('time (s)'); plt.ylabel('vertical velocity (m/s)')
    plt.title('Vertical Velocity vs Time — Euler vs RK4'); plt.legend()
    
    plt.figure()                                         # x vs y plot in space
    plt.plot(xs_r, alt_r)
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.title('x vs y')

    plt.show()
    
def plot_sweep(angles, apogees, ranges):           # sensitivity plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # apogee vs angle
    ax1.plot(angles, apogees, marker='o')
    ax1.set_xlabel('launch angle (deg)')
    ax1.set_ylabel('apogee (m)')
    ax1.set_title('Apogee vs Launch Angle')
    ax1.grid(True, alpha=0.3)

    # range vs angle
    ax2.plot(angles, ranges, marker='o', color='tab:orange')
    ax2.set_xlabel('launch angle (deg)')
    ax2.set_ylabel('range (m)')
    ax2.set_title('Range vs Launch Angle')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Sensitivity to Launch Angle')
    fig.tight_layout()
    
    plt.show()    
    
def plot_histogram(data, label):
    plt.figure()
    plt.hist(data, bins=30)
    plt.axvline(np.mean(data), color='red', linestyle='--', label='mean')
    plt.axvline(np.percentile(data, 5), color='orange', linestyle='--', label='5th')
    plt.axvline(np.percentile(data, 95), color='orange', linestyle='--', label='95')
    plt.xlabel(label)
    plt.ylabel('frequency')
    plt.title('Distribution')
    plt.legend()
    
    plt.show()
    
   
if __name__ == "__main__":
    params["theta_deg"] = 80
    euler = simulate(params, step_euler)       #  run Euler
    rk4   = simulate(params, step_RK4)         #  run RK4
    plot_comparison(euler, rk4)                #  was plot_results(*results) 
    
    angles = np.arange(20, 90, 5)              # sensitivity analysis
    apogees, ranges = angle_ranges(params, angles)  
    plot_sweep(angles, apogees, ranges)
    
    params["theta_deg"] = 80
    apogees, ranges = monte_carlo(params, 500) # montecarlo simulation number
    plot_histogram(apogees, 'apogee (m)')      # apogee historam 
    plot_histogram(ranges, 'range (m)')        # range histogram
    
    
    
    
     
        


    
  
 
    

    
    
    


