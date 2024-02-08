#! /usr/bin/env python
# -*- coding: utf-8 -*-

from pdf_cpu import *
from datetime import datetime
import sys

# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from time import sleep
# from IPython import display
from matplotlib.animation import FuncAnimation
#############################################################################

np.set_printoptions(threshold=sys.maxsize)  # (threshold=np.inf)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

file = open(f'{timestamp}.dat', 'w')
# video_name = "simu"

total_time = 20
anim_fps = 5

N, n_pol, n_theta = 20, 12, 12
ini_grid = create_grid(N, N, n_pol, n_theta)

# When setting up the initial conditions X' and Y' are rotated depending on theta, so when initializing as X and Y
# will actually put the pulse at position X' and Y', which are rotated positions I have to fix this, so I should
# assign the cells at the real lattice and then transfer the allocated data to the self lattice.
ini_grid[int(N / 2.) - 5, int(N / 2.), 3, 1] = 1. / 2.
ini_grid[int(N / 2.) + 5, int(N / 2.), 0, 1] = 1. / 2.

# [y_axis, x_axis, theta, polarization]
# ini_grid[int(N/2.),int(N/2.),0,0] = 1.

# WHY ARE THE CELLS STOPPING AT RANDOM SPOTS????????????
# YOU MUST FIX THE CHANGE OF POLARIZATION DYNAMICS, THAT PASS CELLS WITH ZERO TO ONE IN ALL ORIENTATIONS

fig, ax = plt.subplots()

tissue = Tissue(ini_grid)
tissue.D_theta = 0.1
tissue.D_para_dir = 0.1
tissue.D_perp_dir = 0.1
tissue.kappa = 1.
tissue.gamma = 0.1

tissue.p_max = N / 2.
tissue.c = 1 / 3
tissue.c_d = 10
tissue.D = 0.1

tissue.from_real_lattice()


def print_state(tis):
    aux = np.zeros([tis.n_y, tis.n_x])
    for i in range(0, tis.n_theta):
        for j in range(0, tis.n_pol):
            # aux[:,:] = aux[:,:] + self.lattice[:,:,i,j]
            aux[:, :] = aux[:, :] + tis.real_lattice[:, :, i, j]

    heatmap = ax.pcolormesh(aux, cmap='hot')
    return heatmap


def update(frame):
    tissue.drift_para_dir()
    # Tissue.real_periodic_boundary()
    tissue.diffusion_para_dir()
    # Tissue.real_periodic_boundary()
    tissue.diffusion_perp_dir()
    tissue.real_periodic_boundary()

    tissue.to_real_lattice()
    tissue.diffusion_theta()
    tissue.percolated_diffusion()  # For testing purposes, this function is producing the usual diffusion
    tissue.real_periodic_boundary()
    tissue.polarization_dynamics_dissipation()
    tissue.real_periodic_boundary()
    # Tissue.collision_real_lattice()
    tissue.from_real_lattice()

    tissue.symmetry_break()

    # Tissue.test(time)

    if frame % 1 == 0:
        ax.clear()

    if frame >= total_time:
        anim.event_source.stop()

    stats = str(frame) + " " + str(tissue.total())
    print(stats, file=file)
    # print(stats)

    # heatmap = tissue.print_state()
    heatmap = print_state(tissue)
    return heatmap


anim = FuncAnimation(fig, update, frames=total_time, interval=total_time / anim_fps)
# plt.show()

writer_video = animation.FFMpegWriter(fps=anim_fps)
anim.save(f'{timestamp}.mp4', writer=writer_video)

plt.close('all')
file.close()
