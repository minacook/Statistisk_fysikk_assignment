import numpy as np

r0 = np.array([0,0])
v0 = np.array([0,0])

def create_init_pos(N,R, min_distance = 0.5):
    r_arr = np.zeros((N, 2))
    v_arr = np.zeros((N, 2))
    v_arr[:,1] = 5
    for i in range(N):
        rx = np.random.uniform(-R, R)
        ry = np.random.uniform(-R, R)
        r = np.array([rx, ry])
        r_abs = np.sqrt(rx**2+ry**2)

        d_arr = np.sqrt(np.sum((r_arr - r)**2, axis=1))
        d_min = np.min(d_arr)
        while r_abs > R or d_min < min_distance:
            rx = np.random.uniform(-R,R)
            ry = np.random.uniform(-R,R)
            r_abs = np.sqrt(rx ** 2 + ry ** 2)
        r_arr[i,:] = np.array([rx, ry])

    return r_arr, v_arr

def motion_of_particles(N, R, K, deltat, r_arr, v_arr, T):
    n_steps = T/deltat
    pos = np.zeros((T, N, 2), dtype=float)
    vel = np.zeros((T, N, 2), dtype=float)

    for dt in range(n_steps):



        # Calculating the force
        for i in range(N):
            r_i = np.sqrt(r_arr[i,0]**2+r_arr[i,1]**2)
            r_i_vec = np.array([r_arr[i,0], r_arr[i,1]])
            if r_i > R:
                f_x = -K*(r_i - R) * r_i_vec[0] / r_i
                f_y = -K*(r_i - R) * r_i_vec[1] / r_i
            else:
                f_x, f_y = 0, 0
            for j in range(N):
                if i != j:
                    r_ij_vec = r_i_vec - r_arr[j]  # vector
                    r_ij = np.linalg.norm(r_ij_vec)  # scalar
                    f_x += 24 * (2* (1/r_ij)**12  - (1/r_ij)**6 ) * (r_ij_vec[0]/(r_ij**2))
                    f_y += 24 * (2* (1/r_ij)**12  - (1/r_ij)**6 ) * (r_ij_vec[1]/(r_ij**2))
        f = np.array([f_x, f_y])

