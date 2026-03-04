import numpy as np
import matplotlib.pyplot as plt

def create_init_pos(task, min_distance = 1.1):
    """
    Creates initial positions and velocities for particles within a
    2D circle, where we have v_x = 5 and v_y = 0 for all particles for
    task 2 and random initial velocity for all particles for task 3 and 4.
    The initial positions are distributed so that the min distance between
    particles are 1.1

    """
    r_arr = np.zeros((N, 2))
    r_arr[0, :] = np.array([R - R / 2, 0])

    if task == 'task2': # Creates initial velocities where v_x = 5 and v_y = 0 for all particles
        v_arr = np.zeros((N, 2))
        v_arr[:, 1] = 5
    elif task == 'task3and4': # Creates random initial velocity for all particles
        v_arr = np.random.normal(0, 5.0, size=(N, 2))
        v_arr -= v_arr.mean(axis=0)  # removing COM drift so that <v> = 0

    else:
        print('Task not recognized, must be task2, task3and4')
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

def force_working_on_particles(r_arr):
    """
    Calculates the force working on the particles using the position array with dim(N,2)

    """
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

def motion_of_particles(r_arr, v_arr):
    """
    Calculates the motion of the particles using initial arrays for the position and velocities
    """

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

def calc_energy(pos, vel, t_arr):
    """
    Calculates the energy of the system using position and velocities over time
    """
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

def gas_pressure_from_pos(pos):
    """
    Calculates the pressure in 2D: total wall force / circumference (2 pi R). Returns array with pressure over time
    """
    r = np.linalg.norm(pos, axis=2)  # Getting the absolute pos giving us arr with shape (time, N)
    overlap = np.clip(r - R, 0.0, None)  # Extracting particles that are on the outside (Pushing on the wall)
    Fwall_tot = K * np.sum(overlap, axis=1)
    P = Fwall_tot / (2 * np.pi * R)
    return P

def test_code_const_energy(N_val, R_val, dt_val, K_val, T_val): # Task 2
    """
    Testing that the numerical method is working by checking the total energy over time and plotting it

    """
    global N, R, K, dt, T # Makes variables global so that only the test functions will need to take in the parameters

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

def test_maxwell_distribution(N_val, R_val, dt_val, K_val, T_val): # Task 3

    """
    Verifying the Maxwell distribution for ideal gas and plotting
    the theory vs. collected statistics in a histogram

    """
    global N, R, K, dt, T # Makes variables global so that only the test functions will need to take in the parameters

    N = N_val
    R = R_val
    K = K_val
    dt = dt_val
    T = T_val

    r_init, v_init = create_init_pos('task3and4')
    pos, vel, t_arr = motion_of_particles(r_init, v_init)

    burn = int(0.4 * len(t_arr)) # Using this constant to ignoring the warmup time (approx 40%)
    vx_arr = vel[burn::5, :, 0].ravel() # Getting the vx values, using a stride to avoid correlation and using ravel to get a 1D array for the histogram

    x = np.linspace(vx_arr.min(), vx_arr.max(), 400)

    kBT_from_vx = np.mean(vx_arr ** 2)

    E, Kt, Vwall, Vlj = calc_energy(pos, vel, t_arr)
    kBT_from_K = np.mean(Kt[burn:]) / N

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
        0.02, 0.78, info_text,
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


test_maxwell_distribution(30, 60, 0.005,25, 30)

def test_gas_pressure_a(N_val, R_val, dt_val, K_val, T_val):
    """
    Testing that the gas pressure fulfills ideal gas law
    and plotting the results
    """
    global N, R, K, dt, T
    N, R, dt, K, T = N_val, R_val, dt_val, K_val, T_val

    r_init, v_init = create_init_pos('task3and4')
    pos, vel, t_arr = motion_of_particles(r_init, v_init)

    burn = int(0.4 * len(t_arr))
    stride = 5

    # Pressure time series and mean pressure
    P_arr = gas_pressure_from_pos(pos)           # shape (time,)
    P_mean = np.mean(P_arr[burn::stride])

    # Temperature estimate (use same stride as pressure, for consistency)
    vx_arr = vel[burn::stride, :, 0].ravel()
    kBT = np.mean(vx_arr**2)                     # m=1, kB=1 units

    A = np.pi * R**2
    lhs = P_mean * A
    rhs = N * kBT

    print("⟨P⟩ =", P_mean)
    print("⟨P⟩A =", lhs)
    print("N kBT =", rhs)
    print("Relativ forskjell:", abs(lhs - rhs)/rhs)

    # Plot pressure fluctuations + mean
    fig, ax = plt.subplots()

    ax.plot(t_arr, P_arr, label="P(t)", color='mediumseagreen')
    ax.axhline(P_mean, label="⟨P⟩", color='cornflowerblue')
    ax.axhline(rhs/A, linestyle="--", color='tomato', label=r"\frac{N$k_b$T}{A}")
    info_text = (
        rf"$\langle P \rangle$ = {P_mean:.5f}" "\n"
        rf"$\langle P \rangle A$ = {lhs:.3f}" "\n"
        rf"$N k_B T$ = {rhs:.3f}" "\n"
        rf"Relative difference = {abs(lhs - rhs) / rhs:.2%}"
    )

    ax.text(
        0.02, 0.95,
        info_text,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    ax.set_xlabel("t")
    ax.set_ylabel("Pressure")
    ax.set_title(rf"Pressure vs time (N={N}, R={R}, dt={dt}, K={K}, T={T})")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.show()

def test_gas_pressure_b(N_val, R_val, dt_val, K_val, T_val, n_s):
    """
    Testing how low kinetic initial energy affects the verification of ideal gas law
    and plotting the results in a bar chart
    """
    global N, R, K, dt, T
    N, R, dt, K, T = N_val, R_val, dt_val, K_val, T_val

    ratio = np.zeros(n_s)
    scale = np.linspace(0.5, 0.05, n_s)
    r_init, v_init = create_init_pos('task3and4')
    for i in range(n_s):
        if scale[i] < 0.4:
            T = 60
        pos, vel, t_arr = motion_of_particles(r_init, v_init * (scale[i])**2)

        burn = int(0.4 * len(t_arr))
        stride = 5

        P_arr = gas_pressure_from_pos(pos)
        P_mean = np.mean(P_arr[burn::stride])

        vx_arr = vel[burn::stride, :, 0].ravel()
        kBT = np.mean(vx_arr**2)

        A = np.pi * R**2
        lhs = P_mean * A
        rhs = N * kBT

        ratio[i] = lhs/rhs

    fig, ax = plt.subplots()

    ax.bar(scale, ratio, width=0.1, label=r'$\frac{\langle P\rangle A} {N k_B T}$', color='mediumseagreen')
    ax.axhline(1.0, linestyle="--", color="seagreen", linewidth=2)

    ax.set_ylabel("Ratio")
    ax.set_xlabel("Velocity scale factor $s^2$")
    ax.set_title(f"Breakdown of ideal gas law at different initial energies for (N={N}, R={R}, dt={dt}, K={K}, T={T})")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.show()
