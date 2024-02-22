import math
# import torch
import numpy as np
import scipy as sp


# import numba as nb

# from torch import nn
# from numba import njit


def create_grid(n_y, n_x, n_pol, n_theta):
    return np.zeros([n_y, n_x, n_pol, n_theta])


def resample(arr, N):
    aux_arr = []
    for v in np.vsplit(arr, arr.shape[0] // N):
        aux_arr.extend([*np.hsplit(v, arr.shape[0] // N)])
    return np.array(aux_arr)


def wrap(val, max_val):
    return round(val - max_val * math.floor(val / max_val))


def central_pulse(arr, d, p, theta, prob=1):
    n_x = len(arr[0, :, 0, 0])
    n_y = len(arr[:, 0, 0, 0])
    if d % 2 == 0:
        d += 1
    index_arr = np.arange(-(d // 2), (d // 2) + 1)
    for i in index_arr:
        for j in index_arr:
            arr[int(n_y / 2.) + i, int(n_x / 2.) + j, int(p), int(theta)] = prob/(d**2)
    return arr


class Tissue(object):
    def __init__(self, ini_grid):
        self.real_lattice = ini_grid
        self.n_theta = len(self.real_lattice[0, 0, 0, :])
        self.n_pol = len(self.real_lattice[0, 0, :, 0])
        self.n_x = len(self.real_lattice[0, :, 0, 0])
        self.n_y = len(self.real_lattice[:, 0, 0, 0])
        self.lattice = np.zeros([self.n_y, self.n_x, self.n_theta, self.n_pol])

        self.p_max = 0.0  #
        self.p_mean = 0.

        self.c = 0  #
        self.c_d = 0.0  #

        self.kappa = 0.0
        self.gamma = 0.0
        self.D_theta = 0.0
        self.D_para_dir = 0.0
        self.D_perp_dir = 0.0
        self.D = 0.0

        self.max_x = (1. / (np.sqrt(2))) * self.n_x
        self.max_y = (1. / (np.sqrt(2))) * self.n_y
        
        self.is_drift = False
        self.is_diff_para = False
        self.is_diff_perp = False
        self.is_diff_theta = False
        self.is_diff_perc = False
        self.is_pol_dyn = False
        self.is_sym_break = False
        self.is_collision = False

    def __str__(self):
        return (f'n_x: {self.n_x}\n'
                f'n_y: {self.n_y}\n'
                f'n_pol: {self.n_pol}\n'
                f'n_theta: {self.n_theta}\n'
                '\n'
                f'p_max: {self.p_max}\n'
                f'p_mean: {self.p_mean}\n'
                '\n'
                f'c: {self.c}\n'
                f'c_d: {self.c_d}\n'
                '\n'
                f'kappa: {self.kappa}\n'
                f'gamma: {self.gamma}\n'
                f'D_theta: {self.D_theta}\n'
                f'D_para_dir: {self.D_para_dir}\n'
                f'D_perp_dir: {self.D_perp_dir}\n'
                f'D: {self.D}\n'
                '\n'
                f'drift: {self.is_drift}\n'
                f'diff_para: {self.is_diff_perp}\n'
                f'diff_perp: {self.is_diff_para}\n'
                f'diff_theta: {self.is_diff_theta}\n'
                f'diff_perc: {self.is_diff_perc}\n'
                f'pol_dyn: {self.is_pol_dyn}\n'
                f'sym_break: {self.is_sym_break}\n'
                f'collision: {self.is_collision}')

    def diffusion_para_dir(self):
        D = self.D_para_dir
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        kernel = np.array([[0., 0., 0.],
                           [0.5, -1., 0.5],
                           [0., 0., 0.]])

        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                aux[:, :, i, j] = self.lattice[:, :, i, j] + D * sp.signal.convolve2d(self.lattice[:, :, i, j], kernel,
                                                                                      mode="same", boundary="wrap")
        self.lattice = np.copy(aux)
        self.is_diff_para = True

    def diffusion_perp_dir(self):
        D = self.D_perp_dir
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))

        kernel = np.array([[0., 0.5, 0.],
                           [0., -1., 0.],
                           [0., 0.5, 0.]])

        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                aux[:, :, i, j] = self.lattice[:, :, i, j] + D * sp.signal.convolve2d(self.lattice[:, :, i, j], kernel,
                                                                                      mode="same", boundary="wrap")
        self.lattice = np.copy(aux)
        self.is_diff_perp = True

    def drift_para_dir(self):
        # ROLL METHOD (Uses numpy function roll to convect the pulse in a forwards direction)
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                j_aux = ((1 - self.gamma) ** (self.n_pol - (j + 1)) * self.p_max)
                # print(j_aux, int(j_aux))
                self.lattice[:, :, i, j] = np.roll(self.lattice[:, :, i, j], int(j_aux), axis=1)
                # self.lattice[:,:,i,j] = np.roll(self.lattice[:,:,i,j], j+1, axis=1) Introduce velocity dynamics (
                # dissipation, diffusion and from 0 to 1) The "j+1" term is a quick fix, the correct method should be
                # gathering all particles with j=0 and redistributing then in all orientations equally distributed
                # and with j=1 in all of these orientations
        self.is_drift = True
        # STANDARD METHOD (Same results for both methods)
        # aux = np.zeros((self.n_y,self.n_x,self.n_theta,self.n_pol))
        # for i in range(0, self.n_theta):
        #     for j in range(0, self.n_pol):
        #         for y in range(0, self.n_y):
        #             for x in range(0, self.n_x):
        #                 v_drift = int((j)*self.kappa)
        #                 aux[y,x,i,j] = self.lattice[y-v_drift,x,i,j]
        # self.lattice = np.copy(aux)

    def symmetry_break(self):
        #n = 0
        for i in range(0, self.n_theta):
            p_mean = self.p_mean
            for j in range(0, self.n_pol):
                if j == 0:
                    n = np.random.randint(int(1/self.c))
                    #n += 1
                    if n == 0:
                        #j_new = np.mean(self.lattice[:, :, i, :])
                        #i_new = np.random.randint(0, self.n_theta)
                        self.lattice[:, :, i, j] = self.lattice[:, :, i, int(p_mean/self.n_theta)]
                        #n = 0
                    else:
                        pass
        self.is_sym_break = True

    def diffusion_theta(self):
        # ini_grid = create_grid(self.n_y, self.n_x, self.n_pol, self.n_theta)
        D = self.D_theta
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        for j in range(0, self.n_pol):
            for i in range(0, self.n_theta):
                i_min = i - 1
                if i_min < 0:
                    i_min = self.n_theta - 1
                i_plus = i + 1
                if i_plus > self.n_theta - 1:
                    i_plus = 0

                diff = 0.5 * (self.real_lattice[:, :, i_min, j] +
                              self.real_lattice[:, :, i_plus, j] - 2. * self.real_lattice[:, :, i, j])
                aux[:, :, i, j] = self.real_lattice[:, :, i, j] + D * diff

        self.real_lattice = np.copy(aux)
        self.is_diff_theta = True

    def polarization_dynamics_dissipation(self):
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                for x in range(0, self.n_x):
                    for y in range(0, self.n_y):
                        j_diss = int((j - 1))  # self.gamma
                        if j_diss < self.n_pol:
                            aux[y, x, i, j] = self.lattice[y, x, i, j_diss]
                        else:
                            aux[y, x, i, j] = self.lattice[y, x, i, j]
        self.lattice = np.copy(aux)
        self.is_pol_dyn = True

    def to_real_lattice(self):
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        for i in range(0, self.n_theta):
            if self.lattice[:, :, i, :].any():
                for j in range(0, self.n_pol):
                    if self.lattice[:, :, i, j].any():
                        for x in range(0, self.n_x):
                            if self.lattice[:, x, i, j].any():  # == True:
                                for y in range(0, self.n_y):
                                    d_theta = i * (2. * np.pi / self.n_theta) - np.pi

                                    y_real = y - self.n_y / 2.
                                    x_real = x - self.n_x / 2.

                                    x_new = x_real * np.cos(d_theta) - y_real * np.sin(d_theta) + self.n_x / 2.
                                    y_new = x_real * np.sin(d_theta) + y_real * np.cos(d_theta) + self.n_y / 2.

                                    x_new = wrap(x_new, self.n_x - 1)
                                    y_new = wrap(y_new, self.n_y - 1)

                                    aux[y_new, x_new, i, j] += self.lattice[y, x, i, j]
        self.real_lattice = np.copy(aux)

    def from_real_lattice(self):
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        for i in range(0, self.n_theta):
            if self.real_lattice[:, :, i, :].any():
                for j in range(0, self.n_pol):
                    if self.real_lattice[:, :, i, j].any():
                        for x in range(0, self.n_x):
                            if self.real_lattice[:, x, i, j].any():
                                for y in range(0, self.n_y):
                                    d_theta = i * (2. * np.pi / self.n_theta) - np.pi
                                    y_real = y - self.n_y / 2.
                                    x_real = x - self.n_x / 2.

                                    x_new = x_real * np.cos(d_theta) + y_real * np.sin(d_theta) + self.n_x / 2.
                                    y_new = -x_real * np.sin(d_theta) + y_real * np.cos(d_theta) + self.n_y / 2.

                                    # WHY DID I HAVE TO ADD 0.15 TO THESE VARIABLES?????? IF I DO NOT ADD THESE,
                                    # SOME PULSES STOPPING I DO NOT KNOW THE REASON OF WHY THE PULSES STOP
                                    x_new = wrap(x_new + 0.15, self.n_x - 1)
                                    y_new = wrap(y_new + 0.15, self.n_y - 1)

                                    aux[y_new, x_new, i, j] += self.real_lattice[y, x, i, j]
        self.lattice = np.copy(aux)

    # Try to apply the numba function to speed up the process
    def real_periodic_boundary(self):
        self.to_real_lattice()
        for i in range(0, self.n_theta):
            if self.real_lattice[:, :, i, :].any():
                for j in range(0, self.n_pol):
                    if self.real_lattice[:, :, i, j].any():
                        for x in range(0, self.n_x):
                            if self.real_lattice[:, x, i, j].any():
                                for y in range(0, self.n_y):
                                    if x > self.n_x / 2. + self.max_x / 2.:
                                        aux = self.real_lattice[y, x, i, j]
                                        self.real_lattice[y, x, i, j] = 0.
                                        self.real_lattice[y, x - int(self.max_x), i, j] = self.real_lattice[y, x - int(
                                            self.max_x), i, j] + aux
                                    if y > self.n_x / 2. + self.max_y / 2.:
                                        aux = self.real_lattice[y, x, i, j]
                                        self.real_lattice[y, x, i, j] = 0.
                                        self.real_lattice[y - int(self.max_y), x, i, j] = self.real_lattice[y - int(
                                            self.max_y), x, i, j] + aux
                                    if x < self.n_x / 2. - self.max_x / 2.:
                                        aux = self.real_lattice[y, x, i, j]
                                        self.real_lattice[y, x, i, j] = 0.
                                        self.real_lattice[y, x + int(self.max_x), i, j] = self.real_lattice[y, x + int(
                                            self.max_x), i, j] + aux
                                    if y < self.n_x / 2. - self.max_y / 2.:
                                        aux = self.real_lattice[y, x, i, j]
                                        self.real_lattice[y, x, i, j] = 0.
                                        self.real_lattice[y + int(self.max_y), x, i, j] = self.real_lattice[y + int(
                                            self.max_y), x, i, j] + aux
        self.from_real_lattice()

    def collision_real_lattice(self):
        aux = np.copy(self.real_lattice)

        for y in range(0, self.n_y):
            if self.real_lattice[y, :, :, :].any():
                for x in range(0, self.n_x):
                    if self.real_lattice[y, x, :, :].any():
                        for j in range(0, self.n_theta):
                            if self.real_lattice[y, x, j, :].any():
                                for i in range(0, self.n_pol):
                                    trans_factor = 0.
                                    norm_factor = 0.

                                    mean_pn_x = 0.
                                    mean_pn_y = 0.

                                    for m in range(0, self.n_theta):
                                        if self.real_lattice[y, x, m, :].any():
                                            for n in range(0, self.n_pol):
                                                if m != j or n != i:
                                                    trans_factor += self.real_lattice[y, x, j, i] * self.real_lattice[
                                                        y, x, m, n]
                                                    norm_factor += self.real_lattice[y, x, m, n]

                                                    p_n = n  # * self.kappa
                                                    theta_m = m * (2. * np.pi / self.n_theta)
                                                    mean_pn_x += self.real_lattice[y, x, m, n] * p_n * np.cos(theta_m)
                                                    mean_pn_y += self.real_lattice[y, x, m, n] * p_n * np.sin(theta_m)

                                    p_l = i  # * self.kappa

                                    theta_l = i * (2. * np.pi / self.n_theta)
                                    # theta_k = np.arctan2(mean_pn_y, mean_pn_x)  # + np.pi

                                    px_new = 0.5 * (p_l * np.cos(theta_l) + mean_pn_x)
                                    py_new = 0.5 * (p_l * np.sin(theta_l) + mean_pn_y)

                                    theta_new = np.arctan2(py_new, px_new)  # + np.pi

                                    p_new = np.sqrt(px_new ** 2 + py_new ** 2)

                                    j_new = wrap((int(theta_new * (self.n_theta / (2. * math.pi)))), self.n_theta - 1)
                                    l_new = round(p_new)
                                    l_new = self.n_pol - 1 if l_new >= self.n_pol else l_new

                                    # if trans_factor!=0.:
                                    # print(theta_l, theta_k, theta_new, (theta_l+theta_k)/2., trans_factor)

                                    aux[y, x, j_new, l_new] += trans_factor
                                    aux[y, x, j, i] -= trans_factor

        self.real_lattice = np.copy(aux)
        self.is_collision = True

    def percolated_diffusion(self):
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        D = self.D
        c_d = self.c_d

        # Normalize the density to the maximum density of a cell only to calculate the transition probability.
        # Do not use this loop conditional, it is causing probability to leak
        # if (self.real_lattice[y,:,:,:].any() == True):

        for y in range(1, self.n_y - 1):
            # if (self.real_lattice[y,:,:,:].any() == True):
            for x in range(1, self.n_x - 1):
                # if (self.real_lattice[y,x,:,:].any() == True):
                for i in range(0, self.n_theta):
                    # if (self.real_lattice[y,x,i,:].any() == True):
                    for j in range(0, self.n_pol):
                        P_x_out = D * (self.real_lattice[y, x, i, j] * (
                                np.exp(-c_d * self.real_lattice[y, x + 1, i, j]) + np.exp(
                                        -c_d * self.real_lattice[y, x - 1, i, j])))
                        P_x_in = D * (np.exp(-c_d * self.real_lattice[y, x, i, j]) * (
                                self.real_lattice[y, x + 1, i, j] + self.real_lattice[y, x - 1, i, j]))

                        P_y_out = D * (self.real_lattice[y, x, i, j] * (
                                np.exp(-c_d * self.real_lattice[y + 1, x, i, j]) + np.exp(
                                        -c_d * self.real_lattice[y - 1, x, i, j])))
                        P_y_in = D * (np.exp(-c_d * self.real_lattice[y, x, i, j]) * (
                                self.real_lattice[y + 1, x, i, j] + self.real_lattice[y - 1, x, i, j]))

                        aux[y, x, i, j] = self.real_lattice[y, x, i, j] + P_x_in - P_x_out + P_y_in - P_y_out

                        # kernel= np.array([[0.,0.5*P_y_in,0.],
                        # [0.5*P_x_in,-(P_x_out + P_y_out),0.5*P_x_in],
                        # [0.,0.5*P_y_in,0.]])
                        # aux[y,x,i,j] = self.real_lattice[y,x,i,j] + sp.signal.convolve2d(self.real_lattice[:,:,i,j], kernel, mode="same", boundary="wrap")
        self.real_lattice = np.copy(aux)
        self.is_diff_perc = True
        # aux = np.zeros((self.n_y,self.n_x,self.n_theta,self.n_pol))

        # The interactions must be on the real lattice right? But then how do I evolve the convection in the self
        # lattice? Do I create a Metropolis like algorithm? Where there is a probability of acceptance after the.
        # The convection takes place in the self lattice, but it had low probability of doing so in the real lattice
        # due to interactions?

    def total(self):
        prob = np.sum(np.concatenate(self.real_lattice))
        # prob = np.sum(np.concatenate(self.lattice))
        return prob

    def test(self, time):
        for x in range(0, self.n_x):
            for y in range(0, self.n_y):
                for j in range(0, self.n_theta):
                    for i in range(0, self.n_pol):
                        if self.lattice[y, x, j, i] != 0:
                            d_theta = j * (2. * np.pi / self.n_theta) - np.pi

                            y_real = y - self.n_y / 2.
                            x_real = x - self.n_x / 2.

                            x_new = x_real * np.cos(d_theta) - y_real * np.sin(d_theta) + self.n_x / 2.
                            y_new = x_real * np.sin(d_theta) + y_real * np.cos(d_theta) + self.n_y / 2.

                            x_new = wrap(x_new, self.n_x - 1)
                            y_new = wrap(y_new, self.n_y - 1)
                            print(self.lattice[y, x, j, i], self.real_lattice[y_new, x_new, j, i], time)

    def theta2pol(self, theta):
        pos_value = list(np.arange(0, 360, int(360/self.n_pol)))
        if theta in pos_value:
            return int(pos_value.index(theta))
        else:
            raise ValueError(f'Possible values are: {pos_value}')
    
    def pol2theta(self, pol_dir):
        return int(360/self.n_pol) * pol_dir
