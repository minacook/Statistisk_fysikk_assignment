import numpy as np
import matplotlib.pyplot as plt

N=10
R = 10
K = 25
dt = 0.005
T = 5

def create_init_pos(N,R, min_distance = 0.5):
    r_arr = np.zeros((N, 2))
    v_arr = np.zeros((N, 2))
    v_arr[:,1] = 5
    r_arr[0,:] = np.array([R-R/2, 0])
    for i in range(1,N):
        rx = np.random.uniform(-R, R)
        ry = np.random.uniform(-R, R)
        r = np.array([rx, ry])
        r_abs = np.sqrt(rx**2+ry**2)

        d_arr = np.sqrt(np.sum((r_arr[:i] - r)**2, axis=1))
        d_min = np.min(d_arr)
        while r_abs > R or d_min < min_distance:
            rx = np.random.uniform(-R,R)
            ry = np.random.uniform(-R,R)
            r = np.array([rx, ry])
            r_abs = np.sqrt(rx ** 2 + ry ** 2)
            d_arr = np.sqrt(np.sum((r_arr[:i] - r) ** 2, axis=1))
            d_min = np.min(d_arr)
        r_arr[i,:] = np.array([rx, ry])

    return r_arr, v_arr

def motion_of_particles(N, R, K, dt, r_arr, v_arr, T):
    n_steps = int(T/dt)
    pos = np.zeros((n_steps+1, N, 2), dtype=float)
    vel = np.zeros((n_steps+1, N, 2), dtype=float)
    pos[0] = r_arr
    vel[0] = v_arr
    t_arr = np.linspace(0, T, n_steps+1)

    for t in range(1,n_steps+1):
        f = force_working_on_particles(N, R, K, pos[t-1])
        pos[t] = pos[t-1] + vel[t-1]*dt + (1/2) * f * dt**2

        f_p1 = force_working_on_particles(N, R, K, pos[t])
        vel[t] = vel[t-1] + dt*(f + f_p1)/2
    return pos, vel, t_arr


def force_working_on_particles(N, R, K, r_arr):
    f = np.zeros((N,2))
    for i in range(N):
        r_i_vec = r_arr[i]
        r_i = np.sqrt(np.dot(r_i_vec, r_i_vec))
        if r_i > R:
            f_x = -K * (r_i - R) * r_i_vec[0] / r_i
            f_y = -K * (r_i - R) * r_i_vec[1] / r_i
        else:
            f_x, f_y = 0, 0
        for j in range(N):
            if i != j:
                r_ij_vec = r_i_vec - r_arr[j]  # vector
                r_ij = np.linalg.norm(r_ij_vec)  # scalar
                if r_ij < 10**(-12):
                    continue
                f_x += 24 * (2 * (1 / r_ij) ** 12 - (1 / r_ij) ** 6) * (r_ij_vec[0] / (r_ij ** 2))
                f_y += 24 * (2 * (1 / r_ij) ** 12 - (1 / r_ij) ** 6) * (r_ij_vec[1] / (r_ij ** 2))
        f[i] = np.array([f_x, f_y])

    return f

import numpy as np

def calc_tot_energy(pos, vel, t_arr):
    E = np.zeros_like(t_arr, dtype=float)

    Tsteps = len(t_arr)
    N = pos.shape[1]

    for t in range(Tsteps):
        # Kinetisk energi: 1/2 sum_i |v_i|^2  (m=1)
        K_t = 0.5 * np.sum(vel[t, :, 0]**2 + vel[t, :, 1]**2)

        # Veggpotensial: sum_i 1/2 K (r_i - R)^2 for r_i > R
        r = np.linalg.norm(pos[t], axis=1)  # (N,)
        outside = r > R
        V_wall = 0.5 * K * np.sum((r[outside] - R)**2)

        # Lennard-Jones potensial: sum_{i<j} 4[(1/r)^12 - (1/r)^6]
        V_lj = 0.0
        for i in range(N):
            for j in range(i+1, N):
                rij_vec = pos[t, i] - pos[t, j]
                rij = np.linalg.norm(rij_vec)
                if rij < 1e-12:
                    continue
                inv_r6 = (1.0 / rij)**6
                inv_r12 = inv_r6**2
                V_lj += 4.0 * (inv_r12 - inv_r6)

        E[t] = K_t + V_wall + V_lj

    return E


r_init, v_init = create_init_pos(N,R)
pos_arr, vel_arr, t_arr = motion_of_particles(N, R, K, dt, r_init, v_init, T)

E = calc_tot_energy(pos_arr, vel_arr, t_arr)

plt.plot(t_arr, E)
plt.show()