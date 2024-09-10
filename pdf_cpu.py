import math
import numpy as np
import scipy as sp


def create_grid(n_y, n_x, n_theta, n_pol):
    return np.zeros([n_y, n_x, n_theta, n_pol])


def resample(arr, N):
    aux_arr = []
    for v in np.vsplit(arr, arr.shape[0] // N):
        aux_arr.extend([*np.hsplit(v, arr.shape[0] // N)])
    return np.array(aux_arr)


def wrap(val, max_val):
    return round(val - max_val * math.round(val / max_val))


def central_pulse(arr, radius, theta, p, prob):
    n_x = len(arr[0, :, 0, 0])
    n_y = len(arr[:, 0, 0, 0])
    if radius % 2 == 0:
        radius += 1
    index_arr = np.arange(-(radius // 2), (radius // 2) + 1)
    for i in index_arr:
        for j in index_arr:
            arr[int(n_y / 2.) + i, int(n_x / 2.) + j, int(theta), int(p)] = prob / (radius ** 2)
    # print(arr[:,:,theta, p])
    return arr


def collision_pulse(arr, radius, theta, p, prob=1):
    ini_dist = 2
    n_x = len(arr[0, :, 0, 0])
    n_y = len(arr[:, 0, 0, 0])
    n_theta = len(arr[0, 0, :, 0])
    if radius % 2 == 0:
        radius += 1
    index_arr = np.arange(-(radius // 2), (radius // 2) + 1)
    for i in index_arr:
        for j in index_arr:
            arr[int(n_y / 2.) + i, int(n_x / 2.) + j + ini_dist, int(theta), int(p)] = 0.5 * prob / (radius ** 2)

    for i in index_arr:
        for j in index_arr:
            arr[int(n_y / 2.) + i, int(n_x / 2.) + j - ini_dist, int(theta - n_theta / 2), int(p)] = 0.5 * prob / (
                        radius ** 2)
    return arr


class Tissue(object):
    def __init__(self, ini_grid):
        self.real_lattice = ini_grid
        self.n_pol = len(self.real_lattice[0, 0, 0, :])
        self.n_theta = len(self.real_lattice[0, 0, :, 0])
        self.n_x = len(self.real_lattice[0, :, 0, 0])
        self.n_y = len(self.real_lattice[:, 0, 0, 0])
        self.lattice = np.zeros([self.n_y, self.n_x, self.n_theta, self.n_pol])
        self.old_lattice = np.zeros([self.n_y, self.n_x, self.n_theta, self.n_pol])

        self.c = 1  #
        self.c_d = 0.0  #

        self.kappa = 0.0
        self.gamma = 0.1
        self.D_theta = 0.0
        self.D_para_dir = 0.0
        self.D_perp_dir = 0.0
        self.D = 0.0
        self.D_p = 0.1
        self.g = 4 * self.D_p ** 2  # np.var(np.arange(0, int(1/self.c), 1))

        if (self.D_para_dir + self.D_perp_dir + self.D + self.D_p + self.D_theta) / 5 <= 1:
            if self.D_para_dir + self.D_perp_dir <= 1:
                pass
            else:
                raise ValueError('(D_para_dir + D_perp_dir) must be <= 1')
        else:
            raise ValueError('The diffusion coefficients must be <= 1')

        self.real_max_x = self.n_x + int(1 / 2 * (self.n_x - (np.sqrt(2) * self.n_x)))
        self.real_max_y = self.n_y + int(1 / 2 * (self.n_y - (np.sqrt(2) * self.n_y)))

        self.p_max = int((self.n_x + int(1 / 2 * (self.n_x - (np.sqrt(2) * self.n_x)))) / 2)  #
        self.p_mean = np.sqrt(self.g / (2 * self.gamma))  ## sqrt(v_mean^2)

        self.max_x = int((1. / (np.sqrt(2))) * self.n_x)
        self.max_y = int((1. / (np.sqrt(2))) * self.n_y)

        self.is_drift = False
        self.is_diff_para = False
        self.is_diff_perp = False
        self.is_diff_theta = False
        self.is_diff_perc = False
        self.is_diff_p = False
        self.is_pol_dyn = False
        self.is_sym_break = False
        self.is_collision = False

        self.curr_time = 1
        self.y_inst_1 = 0
        self.x_inst_1 = 0

        self.v_mean_val = 0
        self.v_inst_val = 0
        self.v_arr = [self.curr_time, self.v_inst, self.v_mean]

        self.val_x = 0
        self.val_y = 0

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
                f'diff_p: {self.is_diff_p}\n'
                f'pol_dyn: {self.is_pol_dyn}\n'
                f'sym_break: {self.is_sym_break}\n'
                f'collision: {self.is_collision}')

    def diffusion(self):
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        D = self.D

        for y in range(0, self.n_y):
            for x in range(0, self.n_x):
                for i in range(0, self.n_theta):
                    for j in range(0, self.n_pol):
                        x_min = x - 1
                        if x_min < 0:
                            x_min = self.n_x - 1
                        x_plus = x + 1
                        if x_plus > self.n_x - 1:
                            x_plus = 0

                        y_min = y - 1
                        if y_min < 0:
                            y_min = self.n_y - 1
                        y_plus = y + 1
                        if y_plus > self.n_y - 1:
                            y_plus = 0

                        aux[y, x, i, j] = self.lattice[y, x, i, j] + .5 * D * (
                                    self.lattice[y_plus, x, i, j] + self.lattice[y, x_plus, i, j]
                                    + self.lattice[y_min, x, i, j] + self.lattice[y, x_min, i, j]
                                    - 4 * self.lattice[y, x, i, j])
        self.lattice = np.copy(aux)
        self.is_diff_perc = True

    def diffusion_perp(self):
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        D = self.D

        for y in range(0, self.n_y):
            for x in range(0, self.n_x):
                for i in range(0, self.n_theta):
                    for j in range(0, self.n_pol):
                        y_min = y - 1
                        if y_min < 0:
                            y_min = self.n_y - 1
                        y_plus = y + 1
                        if y_plus > self.n_y - 1:
                            y_plus = 0

                        aux[y, :, :, :] = self.lattice[y, :, :, :] + .5 * D * (
                                    self.lattice[y_plus, :, :, :] + self.lattice[y_min, :, :, :] - 2 * self.lattice[y,
                                                                                                       :, :, :])
        self.lattice = np.copy(aux)
        self.is_diff_perc = True

    def diffusion_p(self):
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        D = self.D_p
        for y in range(0, self.n_y):
            for x in range(0, self.n_x):
                for i in range(0, self.n_theta):
                    for j in range(0, self.n_pol):
                        if j == 0:
                            aux[y, x, i, j] = self.real_lattice[y, x, i, j] + .5 * D * (
                                        self.real_lattice[y, x, i, j + 1] - self.real_lattice[y, x, i, j])
                        if j == self.n_pol - 1:
                            aux[y, x, i, j] = self.real_lattice[y, x, i, j] + .5 * D * (
                                        self.real_lattice[y, x, i, j - 1] - self.real_lattice[y, x, i, j])
                        if j > 0 and j < self.n_pol - 1:
                            aux[y, x, i, j] = self.real_lattice[y, x, i, j] + .5 * D * (
                                        self.real_lattice[y, x, i, j + 1] + self.real_lattice[y, x, i, j - 1] - 2 *
                                        self.real_lattice[y, x, i, j])
        self.real_lattice = np.copy(aux)
        self.is_diff_p = True

    def drift_para_dir(self, ax=1):
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                j_aux = ((1 - self.gamma) ** (self.n_pol - (j + 1)) * self.p_max)
                # print(j_aux, int(j_aux))
                self.lattice[:, :, i, j] = np.roll(self.lattice[:, :, i, j], round(j_aux), axis=1)
                # self.lattice[:,:,i,j] = np.roll(self.lattice[:,:,i,j], j+1, axis=1) Introduce velocity dynamics (
                # dissipation, diffusion and from 0 to 1) The "j+1" term is a quick fix, the correct method should be
                # gathering all particles with j=0 and redistributing then in all orientations equally distributed
                # and with j=1 in all of these orientations
        self.is_drift = True
        """aux = np.zeros((self.n_y,self.n_x,self.n_theta,self.n_pol))
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                for y in range(0, self.n_y):
                    for x in range(0, self.n_x):
                        #p_j= int((j)*self.kappa)
                        p_j = int((1-self.gamma)**(self.n_pol-j)*self.p_max)
                        if p_j < 0:
                            p_j = 0
                        aux[y,x,i,j] = self.lattice[y,x-p_j,i,j]
                #print(p_j,' ', self.gamma,' ',self.n_pol,' ',j,' ',self.p_max)
        self.lattice = np.copy(aux)"""

        # ROLL METHOD (Uses numpy function roll to convect the pulse in a forwards direction)
        # for i in range(0, self.n_theta):
        #     for j in range(0, self.n_pol):
        #         p_j = (1-self.gamma)**(self.n_pol-j)*self.p_max
        #         if p_j < 0:
        #             p_j = 0
        # #         self.lattice[:, :, i, j] = np.roll(self.lattice[:, :, i, j], int(p_j), axis=int(ax))
        self.is_drift = True

    def symmetry_break(self):
        # n = 0
        for x in range(0, self.n_x):
            for y in range(0, self.n_x):
                for i in range(0, self.n_theta):
                    rho_n = (self.c) * np.sum(self.real_lattice[y, x, :, 0]) / self.n_theta
                    if self.real_lattice[y, x, i, 0] >= rho_n:
                        self.real_lattice[y, x, i, 0] = self.real_lattice[y, x, i, 0] - rho_n
                        self.real_lattice[y, x, i, int(self.n_pol / 2)] = self.real_lattice[
                                                                              y, x, i, int(self.n_pol / 2)] + rho_n
        self.is_sym_break = True

    def diffusion_theta(self):
        # ini_grid = create_grid(self.n_y, self.n_x, self.n_theta, self.n_pol)
        D = self.D_theta
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        # for j in range(0, self.n_pol):
        for i in range(0, self.n_theta):
            i_min = i - 1
            if i_min < 0:
                i_min = self.n_theta - 1
            i_plus = i + 1
            if i_plus > self.n_theta - 1:
                i_plus = 0
            aux[:, :, i, :] = self.real_lattice[:, :, i, :] + D * 0.5 * (
                        self.real_lattice[:, :, i_min, :] + self.real_lattice[:, :, i_plus, :] - 2. * self.real_lattice[
                                                                                                      :, :, i, :])

        self.real_lattice = np.copy(aux)
        self.is_diff_theta = True

    def polarization_dynamics_dissipation(self):
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                for x in range(0, self.n_x):
                    for y in range(0, self.n_y):
                        gamma = self.g / (2 * (self.n_pol / 4) ** 2)
                        if j == 0:
                            aux[y, x, i, 0] = self.real_lattice[y, x, i, 0] + gamma * self.real_lattice[y, x, i, 1]
                        if j == self.n_pol - 1:
                            aux[y, x, i, self.n_pol - 1] = self.real_lattice[y, x, i, self.n_pol - 1] - gamma * \
                                                           self.real_lattice[y, x, i, self.n_pol - 1]
                        if j != 0 and j != self.n_pol - 1:
                            aux[y, x, i, j] = self.real_lattice[y, x, i, j] + gamma * (
                                        self.real_lattice[y, x, i, j + 1] - self.real_lattice[y, x, i, j])
        self.real_lattice = np.copy(aux)
        self.is_pol_dyn = True

    def to_real_lattice(self):
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        for i in range(0, self.n_theta):
            # if self.lattice[:, :, i, :].any():
            for j in range(0, self.n_pol):
                # if self.lattice[:, :, i, j].any():
                for x in range(0, self.n_x):
                    # if self.lattice[:, x, i, j].any():  # == True:
                    for y in range(0, self.n_y):
                        # d_theta = i * (2. * np.pi / self.n_theta) #- np.pi
                        #
                        # y_real = y - self.n_y / 2.
                        # x_real = x - self.n_x / 2.
                        #
                        # x_new = x_real * np.cos(d_theta) - y_real * np.sin(d_theta) + self.n_x / 2.
                        # y_new = x_real * np.sin(d_theta) + y_real * np.cos(d_theta) + self.n_y / 2.
                        #
                        # if x_new > self.n_x - 1:
                        #     x_new = x_new -(self.n_x-1)
                        # if x_new < 0:
                        #     x_new = x_new +(self.n_x-1)
                        #
                        # if y_new > self.n_y - 1:
                        #     y_new = y_new -(self.n_y-1)
                        # if y_new < 0:
                        #     y_new = y_new +(self.n_y-1)
                        #
                        # x_new = round(x_new)
                        # y_new = round(y_new)
                        #
                        # # x_new = wrap(x_new, self.n_x - 1)
                        # # y_new = wrap(y_new, self.n_y - 1)
                        #
                        # aux[y_new, x_new, i, j] += self.lattice[y, x, i, j]
                        d_theta = i * (2. * np.pi / self.n_theta)

                        y_real = y - self.n_y / 2.
                        x_real = x - self.n_x / 2.

                        x_new = x_real * np.cos(d_theta) - y_real * np.sin(d_theta) + self.n_x / 2.
                        y_new = x_real * np.sin(d_theta) + y_real * np.cos(d_theta) + self.n_y / 2.

                        # Handle x and y wrapping for values outside the grid range
                        x_new = (x_new + self.n_x) % self.n_x
                        y_new = (y_new + self.n_y) % self.n_y

                        # Convert to integers after modulo and ensure they don't exceed bounds
                        x_new = int(round(x_new))
                        y_new = int(round(y_new))

                        # Clamp values to stay within valid indices
                        x_new = min(max(x_new, 0), self.n_x - 1)
                        y_new = min(max(y_new, 0), self.n_y - 1)

                        aux[y_new, x_new, i, j] += self.lattice[y, x, i, j]

        self.real_lattice = np.copy(aux)

    def from_real_lattice(self):
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        for i in range(0, self.n_theta):
            # if self.real_lattice[:, :, i, :].any():
            for j in range(0, self.n_pol):
                # if self.real_lattice[:, :, i, j].any():
                for x in range(0, self.n_x):
                    # if self.real_lattice[:, x, i, j].any():
                    for y in range(0, self.n_y):
                        # d_theta = i * (2. * np.pi / self.n_theta) #- np.pi
                        # y_real = y - self.n_y / 2.
                        # x_real = x - self.n_x / 2.
                        #
                        # x_new = x_real * np.cos(d_theta) + y_real * np.sin(d_theta) + self.n_x / 2.
                        # y_new = -x_real * np.sin(d_theta) + y_real * np.cos(d_theta) + self.n_y / 2.
                        #
                        # if x_new > self.n_x - 1:
                        #     x_new = x_new -(self.n_x-1)
                        # if x_new < 0:
                        #     x_new = x_new +(self.n_x-1)
                        #
                        # if y_new > self.n_y - 1:
                        #     y_new = y_new -(self.n_y-1)
                        # if y_new < 0:
                        #     y_new = y_new +(self.n_y-1)
                        #
                        # x_new = round(x_new)
                        # y_new = round(y_new)
                        #
                        # # WHY DID I HAVE TO ADD 0.15 TO THESE VARIABLES?????? IF I DO NOT ADD THESE,
                        # # SOME PULSES STOPPING I DO NOT KNOW THE REASON OF WHY THE PULSES STOP
                        # # y_new = wrap(y_new + 0.15, self.n_y - 1)
                        # # x_new = wrap(x_new + 0.15, self.n_x - 1)
                        #
                        # aux[y_new, x_new, i, j] += self.real_lattice[y, x, i, j]
                        d_theta = i * (2. * np.pi / self.n_theta)

                        y_real = y - self.n_y / 2.
                        x_real = x - self.n_x / 2.

                        x_new = x_real * np.cos(d_theta) + y_real * np.sin(d_theta) + self.n_x / 2.
                        y_new = -x_real * np.sin(d_theta) + y_real * np.cos(d_theta) + self.n_y / 2.

                        # Handle x and y wrapping for values outside the grid range
                        x_new = (x_new + self.n_x) % self.n_x
                        y_new = (y_new + self.n_y) % self.n_y

                        # Convert to integers after modulo and ensure they don't exceed bounds
                        x_new = int(round(x_new))
                        y_new = int(round(y_new))

                        # Clamp values to stay within valid indices
                        x_new = min(max(x_new, 0), self.n_x - 1)
                        y_new = min(max(y_new, 0), self.n_y - 1)

                        aux[y_new, x_new, i, j] += self.real_lattice[y, x, i, j]

        self.lattice = np.copy(aux)

    # Try to apply the numba function to speed up the process
    def real_periodic_boundary(self):
        # self.to_real_lattice()
        # for i in range(0, self.n_theta):
        #     # if self.real_lattice[:, :, i, :].any():
        #     for j in range(0, self.n_pol):
        #         # if self.real_lattice[:, :, i, j].any():
        #         for x in range(0, self.n_x):
        #             # if self.real_lattice[:, x, i, j].any():
        #             for y in range(0, self.n_y):
        #                 if x > round(self.n_x / 2.) + round(self.max_x / 2.):
        #                     aux = np.copy(self.real_lattice[y, x, i, j])
        #                     self.real_lattice[y, x, i, j] = 0.
        #                     self.real_lattice[y, x - round(self.max_x), i, j] = self.real_lattice[y, x - round(self.max_x), i, j] + aux
        #                 if y > round(self.n_x / 2.) + round(self.max_y / 2.):
        #                     aux = np.copy(self.real_lattice[y, x, i, j])
        #                     self.real_lattice[y, x, i, j] = 0.
        #                     self.real_lattice[y - round(self.max_y), x, i, j] = self.real_lattice[y - round(self.max_y), x, i, j] + aux
        #                 if x < round(self.n_x / 2.) - round(self.max_x / 2.):
        #                     aux = np.copy(self.real_lattice[y, x, i, j])
        #                     self.real_lattice[y, x, i, j] = 0.
        #                     self.real_lattice[y, x + round(self.max_x), i, j] = self.real_lattice[y, x + round(self.max_x), i, j] + aux
        #                 if y < round(self.n_x / 2.) - round(self.max_y / 2.):
        #                     aux = np.copy(self.real_lattice[y, x, i, j])
        #                     self.real_lattice[y, x, i, j] = 0.
        #                     self.real_lattice[y + round(self.max_y), x, i, j] = self.real_lattice[y + round(self.max_y), x, i, j] + aux
        # Iterate over the grid points
        for i in range(self.n_theta):
            for j in range(self.n_pol):
                for x in range(self.n_x):
                    for y in range(self.n_y):
                        # Periodic boundary in the x-direction (left to right and right to left)
                        if x >= (self.n_x / 2.) + (self.max_x / 2.):
                            new_x = (x - self.max_x - 1) % self.n_x
                            self.real_lattice[y, new_x, i, j] += self.real_lattice[y, x, i, j]
                            self.real_lattice[y, x, i, j] = 0
                        elif x < (self.n_x / 2.) - (self.max_x / 2.):
                            new_x = (x + self.max_x) % self.n_x
                            self.real_lattice[y, new_x, i, j] += self.real_lattice[y, x, i, j]
                            self.real_lattice[y, x, i, j] = 0

                        # Periodic boundary in the y-direction (top to bottom and bottom to top)
                        if y >= (self.n_y / 2.) + (self.max_y / 2.):
                            new_y = (y - self.max_y - 1) % self.n_y
                            self.real_lattice[new_y, x, i, j] += self.real_lattice[y, x, i, j]
                            self.real_lattice[y, x, i, j] = 0
                        elif y < (self.n_y / 2.) - (self.max_y / 2.):
                            new_y = (y + self.max_y) % self.n_y
                            self.real_lattice[new_y, x, i, j] += self.real_lattice[y, x, i, j]
                            self.real_lattice[y, x, i, j] = 0

        # print(self.n_x / 2, self.max_x / 2)
        # self.from_real_lattice()

    def collision_real_lattice(self):
        aux = np.copy(self.real_lattice)

        for y in range(0, self.n_y):
            if (self.real_lattice[y, :, :, :].any() == True):
                for x in range(0, self.n_x):
                    if (self.real_lattice[y, x, :, :].any() == True):
                        for j in range(0, self.n_theta):
                            if (self.real_lattice[y, x, j, :].any() == True):
                                for l in range(0, self.n_pol):
                                    trans_factor = 0.
                                    norm_factor = 0.

                                    mean_pn_x = 0.
                                    mean_pn_y = 0.

                                    for m in range(0, self.n_theta):
                                        if (self.real_lattice[y, x, m, :].any() == True):
                                            for n in range(0, self.n_pol):
                                                if m != j or n != l:
                                                    trans_factor += self.real_lattice[y, x, j, l] * self.real_lattice[
                                                        y, x, m, n]
                                                    norm_factor += self.real_lattice[y, x, m, n]

                                                    p_n = (n)  # * self.kappa
                                                    theta_m = m * (2. * np.pi / self.n_theta)
                                                    mean_pn_x += self.real_lattice[y, x, m, n] * p_n * np.cos(theta_m)
                                                    mean_pn_y += self.real_lattice[y, x, m, n] * p_n * np.sin(theta_m)

                                    p_l = l  # * self.kappa

                                    theta_l = j * (2. * np.pi / self.n_theta)
                                    theta_k = np.arctan2(mean_pn_y, mean_pn_x)  # + np.pi

                                    px_new = 0.5 * (p_l * np.cos(theta_l) + mean_pn_x)
                                    py_new = 0.5 * (p_l * np.sin(theta_l) + mean_pn_y)

                                    theta_new = np.arctan2(py_new, px_new)  # + np.pi

                                    p_new = np.sqrt(px_new * 2 + py_new * 2)

                                    j_new = self.wrap((int(theta_new * (self.n_theta / (2. * math.pi)))),
                                                      self.n_theta - 1)
                                    l_new = round(p_new)
                                    l_new = self.n_pol - 1 if l_new >= self.n_pol else l_new

                                    # if trans_factor!=0.:
                                    # print(theta_l, theta_k, theta_new, (theta_l+theta_k)/2., trans_factor)

                                    aux[y, x, j_new, l_new] += trans_factor
                                    aux[y, x, j, l] -= trans_factor

        self.real_lattice = np.copy(aux)
        self.is_collision = True

    def percolated_diffusion(self):
        aux = np.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        D = 0.5  # self.D
        c_d = self.c_d
        #
        # for y in range(1, self.n_y - 1):
        #     # if (self.real_lattice[y,:,:,:].any() == True):
        #     for x in range(1, self.n_x - 1):
        #         # if (self.real_lattice[y,x,:,:].any() == True):
        #         for i in range(0, self.n_theta):
        #             # if (self.real_lattice[y,x,i,:].any() == True):
        #             for j in range(0, self.n_pol):
        #                 P_x_out = D*( (self.real_lattice[y,x,i,j]) * ( (1 - self.real_lattice[y,x+1,i,j]) + (1 - self.real_lattice[y,x-1,i,j]) )) * self.real_lattice[y,x,i,j]
        #                 P_x_in = D*((1 - self.real_lattice[y,x,i,j]) * ( self.real_lattice[y,x+1,i,j]*self.real_lattice[y,x+1,i,j] + self.real_lattice[y,x-1,i,j]*self.real_lattice[y,x-1,i,j] ))
        #
        #                 P_y_out = D*( (self.real_lattice[y,x,i,j]) * ( (1 - self.real_lattice[y+1,x,i,j]) + (1 - self.real_lattice[y-1,x,i,j]) )) * self.real_lattice[y,x,i,j]
        #                 P_y_in = D*((1 - self.real_lattice[y,x,i,j]) * ( self.real_lattice[y+1,x,i,j]*self.real_lattice[y+1,x,i,j] + self.real_lattice[y-1,x,i,j]*self.real_lattice[y-1,x,i,j] ))
        #
        #                 aux[y, x, i, j] = self.real_lattice[y, x, i, j] + P_x_in - P_x_out + P_y_in - P_y_out

        # Test, make function that is either 0 or 1, then it rejects when there the next cell is different than zero
        for y in range(1, self.n_y - 1):
            # if (self.real_lattice[y,:,:,:].any() == True):
            for x in range(1, self.n_x - 1):
                # if (self.real_lattice[y,x,:,:].any() == True):
                for i in range(0, self.n_theta):
                    # if (self.real_lattice[y,x,i,:].any() == True):
                    for j in range(0, self.n_pol):
                        P_x_out = D * ((self.real_lattice[y, x, i, j] ** 2) * (
                                    (1.0 - self.real_lattice[y, x + 1, i, j]) + (
                                        1.0 - self.real_lattice[y, x - 1, i, j])))
                        P_x_in = D * ((1.0 - self.real_lattice[y, x, i, j]) * (
                                    self.real_lattice[y, x + 1, i, j] ** 2 + self.real_lattice[y, x - 1, i, j] ** 2))

                        P_y_out = D * ((self.real_lattice[y, x, i, j] ** 2) * (
                                    (1.0 - self.real_lattice[y + 1, x, i, j]) + (
                                        1.0 - self.real_lattice[y - 1, x, i, j])))
                        P_y_in = D * ((1.0 - self.real_lattice[y, x, i, j]) * (
                                    self.real_lattice[y + 1, x, i, j] ** 2 + self.real_lattice[y - 1, x, i, j] ** 2))

                        aux[y, x, i, j] = self.real_lattice[y, x, i, j] + P_x_in - P_x_out + P_y_in - P_y_out

        self.real_lattice = np.copy(aux)
        self.is_diff_perc = True

    def total(self):
        prob = np.sum(np.concatenate(self.real_lattice))
        # prob = np.sum(np.concatenate(self.lattice))
        return prob

    def test(self, time):
        for x in range(0, self.n_x):
            for y in range(0, self.n_y):
                for i in range(0, self.n_theta):
                    for j in range(0, self.n_pol):
                        if self.lattice[y, x, i, j] != 0:
                            d_theta = j * (2. * np.pi / self.n_theta) - np.pi

                            y_real = y - self.n_y / 2.
                            x_real = x - self.n_x / 2.

                            x_new = x_real * np.cos(d_theta) - y_real * np.sin(d_theta) + self.n_x / 2.
                            y_new = x_real * np.sin(d_theta) + y_real * np.cos(d_theta) + self.n_y / 2.

                            x_new = wrap(x_new, self.n_x - 1)
                            y_new = wrap(y_new, self.n_y - 1)
                            print(self.lattice[y, x, i, j], self.real_lattice[y_new, x_new, i, j], time)

    def msd(self):
        sum = 0
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                for x in range(0, self.n_x):
                    for y in range(0, self.n_y):
                        sum += ((x - self.n_x / 2) ** 2 + (y - self.n_y / 2) ** 2) * self.real_lattice[y, x, i, j]
        return (sum)

    def meann(self):
        sum_x = 0
        sum_y = 0
        sum_2x = 0
        sum_2y = 0
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                for x in range(0, self.n_x):
                    for y in range(0, self.n_y):
                        sum_x += x * self.real_lattice[y, x, i, j]
                        sum_y += y * self.real_lattice[y, x, i, j]
                        sum_2x += (x ** 2) * self.real_lattice[y, x, i, j]
                        sum_2y += (y ** 2) * self.real_lattice[y, x, i, j]
        return (sum_x, sum_2x, sum_y, sum_2y)  # (sum_x, sum_2x-sum_x**2, sum_y, sum_2y-sum_y**2)

    def theta2pol(self, theta):
        pos_value = list(np.arange(0, 360, round(360 / self.n_pol)))
        if theta in pos_value:
            return round(pos_value.index(theta))
        else:
            raise ValueError(f'Possible values are: {pos_value}')

    def pol2theta(self, pol_dir):
        return round(360 / self.n_pol) * pol_dir

    def max_val(self, theta, p):
        rho = []
        rho_index = []
        for x in range(self.n_x):
            for y in range(self.n_y):
                rho.append(self.real_lattice[y, x, theta, p])  # (np.max(self.real_lattice[y, x, theta, p]))
                rho_index.append([y, x, theta, p])  # (np.argmax(self.real_lattice[y, x, theta, p]))
                # print(f'v = {np.argmax(self.real_lattice[y, :, i, j])}, {np.max(self.real_lattice[y, :, i, j])}')
        return [np.max(rho), rho_index[rho.index(np.max(rho))]]

    def v_mean(self, theta, p, ori: list):
        sqrt2 = np.sqrt(2)
        self.val_x, self.val_y = abs(round(1 / 2 * (self.n_x - (sqrt2 * self.n_x)))), abs(
            round(1 / 2 * (self.n_y - (sqrt2 * self.n_y))))
        v_c_x, v_c_y = 1, 1
        print(self.val_x, self.val_y)
        if len(ori) != 2:
            raise AssertionError('ori array must have size 2')
        y_0 = round(ori[0])
        x_0 = round(ori[1])
        # print(self.curr_time)
        # self.v_mean_val = np.sqrt((self.max_val(theta, p)[1][1]-x_0)**2 + (self.max_val(theta, p)[1][0]-y_0)**2)/self.curr_time
        if self.max_val(theta, p)[1][0] >= self.real_max_y:
            self.v_mean_val = np.sqrt(
                (self.max_val(theta, p)[1][1] - x_0) ** 2 + (self.max_val(theta, p)[1][0] - y_0) ** 2) / self.curr_time
            print(f'yyy = {self.max_val(theta, p)[1][0]}, {self.n_y}')
            self.val_y += 1
        elif self.max_val(theta, p)[1][1] >= self.real_max_x:
            self.v_mean_val = np.sqrt((self.max_val(theta, p)[1][1] + self.val_x - x_0) ** 2 + (
                        self.max_val(theta, p)[1][0] - y_0) ** 2) / self.curr_time
            print(f'yyy = {self.max_val(theta, p)[1][1]}, {self.n_x}')
        else:
            self.v_mean_val = np.sqrt(
                (self.max_val(theta, p)[1][1] - x_0) ** 2 + (self.max_val(theta, p)[1][0] - y_0) ** 2) / self.curr_time
            self.val_x += 1
        print(f'x={self.max_val(theta, p)[1][0]}, y={self.max_val(theta, p)[1][1]}. v={self.v_mean_val}')
        return self.v_mean_val, round(self.v_mean_val)

    def v_inst(self, theta, p):
        y_curr = self.max_val(theta, p)[1][0]
        x_curr = self.max_val(theta, p)[1][1]
        self.v_inst_val = np.sqrt((x_curr - self.x_inst_1) ** 2 + (y_curr - self.y_inst_1) ** 2)
        return self.v_inst_val, round(self.v_inst_val)

    def up_max_xy(self, theta, p):
        self.y_inst_1 = self.max_val(theta, p)[1][0]
        self.x_inst_1 = self.max_val(theta, p)[1][1]

    def var_d(self, d):
        return 4 * (d ** 2)
