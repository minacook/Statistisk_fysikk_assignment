import numpy as np
import matplotlib.pyplot as plt

def create_init_pos(task, min_distance = 1):
    r_arr = np.zeros((N, 2))
    r_arr[0, :] = np.array([R - R / 2, 0])

    if task == 'task2':
        v_arr = np.zeros((N, 2))
        v_arr[:, 1] = 5
    elif task == 'task3':
        v_arr = np.random.normal(0, 5.0, size=(N, 2))
        v_arr -= v_arr.mean(axis=0)  # removing COM drift so that <v> = 0
    else:
        print('Task not recognized, must be task2 or task3')
        v_arr = None

    for i in range(1,N):
        rx = np.random.uniform(-R, R)
        ry = np.random.uniform(-R, R)
        r = np.array([rx, ry])
        r_abs = np.sqrt(rx**2+ry**2)

        d_arr = np.sqrt(np.sum((r_arr[:i] - r)**2, axis=1))
        d_min = np.min(d_arr)
        while r_abs >= R or d_min < min_distance:
            rx = np.random.uniform(-R,R)
            ry = np.random.uniform(-R,R)
            r = np.array([rx, ry])
            r_abs = np.sqrt(rx ** 2 + ry ** 2)
            d_arr = np.sqrt(np.sum((r_arr[:i] - r) ** 2, axis=1))
            d_min = np.min(d_arr)
        r_arr[i,:] = np.array([rx, ry])

    return r_arr, v_arr

def motion_of_particles(r_arr, v_arr):
    n_steps = int(T/dt)
    pos = np.zeros((n_steps+1, N, 2), dtype=float)
    vel = np.zeros((n_steps+1, N, 2), dtype=float)
    pos[0] = r_arr
    vel[0] = v_arr
    t_arr = np.linspace(0, T, n_steps+1)

    for t in range(1,n_steps+1):
        f = force_working_on_particles(pos[t-1])
        pos[t] = pos[t-1] + vel[t-1]*dt + (1/2) * f * dt**2

        f_p1 = force_working_on_particles(pos[t])
        vel[t] = vel[t-1] + dt*(f + f_p1)/2
    return pos, vel, t_arr


def force_working_on_particles(r_arr):
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

'''def calc_tot_energy(pos, vel, t_arr):
    E = np.zeros_like(t_arr, dtype=float)

    Tsteps = len(t_arr)
    N = pos.shape[1]

    for t in range(Tsteps):
        K_t = 0.5 * np.sum(vel[t, :, 0]**2 + vel[t, :, 1]**2)

        r = np.linalg.norm(pos[t], axis=1)
        outside = r > R
        V_wall = 0.5 * K * np.sum((r[outside] - R)**2)

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

    return E'''

def calc_energy(pos, vel, t_arr):
    E = np.zeros_like(t_arr, dtype=float)
    Kt = np.zeros_like(t_arr, dtype=float)
    Vwall = np.zeros_like(t_arr, dtype=float)
    Vlj = np.zeros_like(t_arr, dtype=float)

    Tsteps = len(t_arr)
    N = pos.shape[1]

    for t in range(Tsteps):
        Kt[t] = 0.5 * np.sum(vel[t, :, 0]**2 + vel[t, :, 1]**2)

        r = np.linalg.norm(pos[t], axis=1)
        outside = r > R
        Vwall[t] = 0.5 * K * np.sum((r[outside] - R)**2)

        V = 0.0
        for i in range(N):
            for j in range(i+1, N):
                rij = np.linalg.norm(pos[t, i] - pos[t, j])
                if rij < 1e-12:
                    continue
                inv_r6 = (1.0 / rij)**6
                inv_r12 = inv_r6**2
                V += 4.0 * (inv_r12 - inv_r6)

        Vlj[t] = V
        E[t] = Kt[t] + Vwall[t] + Vlj[t]

    return E, Kt, Vwall, Vlj

def test_code_const_energy(N_val, R_val, dt_val, K_val, T_val):
    global N, R, K, dt, T

    N = N_val
    R = R_val
    K = K_val
    dt = dt_val
    T = T_val

    r_init, v_init = create_init_pos('task2')
    pos_arr, vel_arr, t_arr = motion_of_particles(r_init, v_init)
    E, _, _, _ = calc_energy(pos_arr, vel_arr, t_arr)

    plt.plot(t_arr, E, label=f'N={N}, R={R}, dt={dt}', color = 'mediumseagreen')
    plt.xlabel('t'); plt.ylabel('E'); plt.grid(alpha=0.2)
    plt.title('Energy over time')
    plt.legend()
    plt.show()

N1, R1, dt1 = 10, 10, 0.005
N2, R2, dt2 = 40, 50, 0.005


def test_maxwell_distribution(N_val, R_val, dt_val, K_val, T_val):
    global N, R, K, dt, T

    N = N_val
    R = R_val
    K = K_val
    dt = dt_val
    T = T_val

    r_init, v_init = create_init_pos('task3')
    pos, vel, t_arr = motion_of_particles(r_init, v_init)

    burn = int(0.4 * len(t_arr)) # Using this constant to ignoring the approximate warmup time
    vx_arr = vel[burn::5, :, 0].ravel() # Getting the vx values, using a stride to avoid correlation and using ravel to get a 1D array for the histogram

    x = np.linspace(vx_arr.min(), vx_arr.max(), 400)

    kBT_from_vx = np.mean(vx_arr ** 2)  # m=1

    E, Kt, Vwall, Vlj = calc_energy(pos, vel, t_arr)
    kBT_from_K = np.mean(Kt[burn:]) / N  # 2D: <K> = N kBT

    rel_diff = abs(kBT_from_vx - kBT_from_K) / kBT_from_K

    prefactor = np.sqrt(1 / (2 * np.pi * kBT_from_vx))
    theory = prefactor * np.exp(-x ** 2 / (2 * kBT_from_vx))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.25)

    # -------- Maxwell --------
    ax[0].hist(vx_arr, bins = 'fd', density=True, alpha=0.5,
               label=rf"$v_x$  (N={N}, R={R}, dt={dt}, K={K}, T={T})",
               color='cornflowerblue')

    ax[0].plot(x, theory, color='tomato', linewidth=2,
               label=r'Gaussian $v_x$')

    ax[0].set_xlabel(r'$v_x$')
    ax[0].set_ylabel('Probability density')
    ax[0].set_title(r'Maxwell verification')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # -------- Energy--------

    info_text = (
        rf"$k_BT$ from $\langle v_x^2 \rangle$ = {kBT_from_vx:.3f}" "\n"
        rf"$k_BT$ from $\langle K \rangle / N$ = {kBT_from_K:.3f}" "\n"
        rf"Relative difference = {rel_diff:.2%}"
    )

    ax[0].text(
        0.02, 0.78, info_text,  # litt lavere enn før
        transform=ax[0].transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )
    ax[1].plot(t_arr, Kt, label='Kinetic Energy')
    ax[1].plot(t_arr, Vlj + Vwall, label='Potential Energy')
    ax[1].plot(t_arr, E, linestyle='--', linewidth=2, label='Total Energy')

    ax[1].set_xlabel('t')
    ax[1].set_ylabel('Energy')
    ax[1].set_title('Energy vs Time')
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

test_maxwell_distribution(40,80,0.001,25,40)