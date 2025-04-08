import numpy as np


def cross_matrix(a):  # 叉乘矩阵
    a_x = np.array([
        [0, -float(a[2]), float(a[1])],
        [float(a[2]), 0, -float(a[0])],
        [-float(a[1]), float(a[0]), 0]
    ])
    return a_x


def d_q(q, omega):  # 四元数的导数
    omega_matrix = np.array([
        [0, -float(omega[0][0]), -float(omega[1][0]), -float(omega[2][0])],
        [float(omega[0][0]), 0, float(omega[2][0]), -float(omega[1][0])],
        [float(omega[1][0]), -float(omega[2][0]), 0, float(omega[0][0])],
        [float(omega[2][0]), float(omega[1][0]), -float(omega[0][0]), 0]
    ])
    return 0.5 * omega_matrix @ q


def d_omega(j_inv, omega, j, u):  # 角加速度
    D_omega = j_inv @ ((-cross_matrix(omega) @ j @ omega) + u)
    return D_omega


def get_q_e(q_d, q):  # 误差四元数
    q_dv = np.array([
        [float(q_d[1][0])],
        [float(q_d[2][0])],
        [float(q_d[3][0])]
    ])
    q_d0 = float(q_d[0][0])
    matiq_d = np.block([
        [q_d.T],
        [-q_dv, q_d0 * np.eye(3) - cross_matrix(q_dv)]
    ])
    qe = matiq_d @ q
    return qe


def get_omega_e(omega, omega_d):  # 误差角速度
    omega_e = omega - omega_d
    return omega_e


def R_K(q, omega, tau, j_inv, j, u):
    K_21 = d_omega(j_inv, omega, j, u)
    K_22 = d_omega(j_inv, omega + K_21 / 2 * tau, j, u)
    K_23 = d_omega(j_inv, omega + K_22 / 2 * tau, j, u)
    K_24 = d_omega(j_inv, omega + K_23 * tau, j, u)
    omega = omega + (K_21 + 2 * K_22 + 2 * K_23 + K_24) / 6 * tau

    K_11 = d_q(q, omega)
    K_12 = d_q(q + K_11 / 2 * tau, omega)
    K_13 = d_q(q + K_12 / 2 * tau, omega)
    K_14 = d_q(q + K_13 * tau, omega)
    q = q + (K_11 + 2 * K_12 + 2 * K_13 + K_14) / 6 * tau

    return q, omega


def get_omega_d(t):
    omega_d = np.array([
        [0],
        [0],
        [0]
    ])
    return omega_d
