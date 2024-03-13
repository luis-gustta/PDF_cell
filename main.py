#! /usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import sys
import os
import math

# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from time import sleep
# from IPython import display
from matplotlib.animation import FuncAnimation
#############################################################################

np.set_printoptions(threshold=sys.maxsize)  # (threshold=np.inf)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# video_name = "simu"
os.mkdir(f'{timestamp}')
os.mkdir(f'{timestamp}/frames')

file = open(f'{timestamp}/{timestamp}.dat', 'w')
sumj_data = open(f'{timestamp}/{timestamp}_sumj.dat', 'w')
info = open(f'{timestamp}/{timestamp}.txt', 'w')

total_time = 200
anim_fps = 5

N, n_pol, n_theta = 501, 12, 12 ## np = 50
ini_grid = create_grid(N, N, n_theta, n_pol)

radius = 1
pol = 0
theta = 6

central_pulse(ini_grid, radius, theta, pol)

fig, ax = plt.subplots()
ax.set_aspect('equal')

tissue = Tissue(ini_grid)
tissue.D_theta = 0.1##0.2#0.05
tissue.D_para_dir = 0.1
tissue.D_perp_dir = 0.1
tissue.kappa = 1.
tissue.gamma = 0.11#9#0.18#1#0.29

tissue.p_max = 4#N / 2.

tissue.c = 1 / 2
tissue.c_d = 10
tissue.D = 0.1
tissue.D_p = 0.4

tissue.from_real_lattice()

sum_j = np.zeros(n_pol)
mean_j = np.zeros(total_time)
var_j = np.zeros(total_time)

mean_theta = np.zeros(total_time+5)
mean_theta_sq = np.zeros(total_time+5)
var_theta = np.zeros(total_time+5)

def print_state(tis):
    aux = np.zeros([tis.n_y, tis.n_x])
    for i in range(0, tis.n_theta):
        for j in range(0, tis.n_pol):
            # aux[:,:] = aux[:,:] + self.lattice[:,:,i,j]
            aux[:, :] = aux[:, :] + tis.real_lattice[:, :, i, j]

    heatmap = ax.pcolormesh(aux, cmap='hot')
    return heatmap

msd_list = []
listt = []

def update(frame):

    tissue.up_max_xy(theta, pol)
    tissue.drift_para_dir()

    tissue.to_real_lattice()
    tissue.real_periodic_boundary()
    tissue.from_real_lattice()

    tissue.diffusion()

    tissue.to_real_lattice()
    tissue.real_periodic_boundary()

    tissue.diffusion_p()
    tissue.symmetry_break()
    tissue.diffusion_theta()
    tissue.polarization_dynamics_dissipation()

    ###tissue.real_periodic_boundary()

    msd_n = tissue.msd()
    msd_list.append(msd_n)
    listt.append(tissue.meann())

    for j in range(int(tissue.n_pol)):
        sum = np.sum(tissue.real_lattice[:, :, :, j])
        sum_j[j] = sum

    for i in range(int(tissue.n_theta)):
        mean_theta[int(tissue.curr_time)] += (np.sum(tissue.real_lattice[:, :, i, :]) * i)#/tissue.n_theta
        mean_theta_sq[int(tissue.curr_time)] += (np.sum(tissue.real_lattice[:, :, i, :]) * i**2)

    var_theta[int(tissue.curr_time)] = mean_theta_sq[[int(tissue.curr_time)]] - mean_theta[[int(tissue.curr_time)]]**2
    tissue.from_real_lattice()
    tissue.curr_time += 1

    if frame % 1 == 0:
        ax.clear()

    if frame >= total_time:
        anim.event_source.stop()

    sumx, sumx2, sumy, sumy2 = tissue.meann()
    total = tissue.total()
    stats = f'{frame} {total} {sumx} {sumx2} {sumy} {sumy2} {msd_n}'
    print(stats, file=file)
    print(sum_j, file=sumj_data)

    heatmap = print_state(tissue)
    plt.title(f'$t=${frame}, $P=$'+"%.8f" % total)
    plt.savefig(f'{timestamp}/frames/{frame}.png')
    plt.title(f'$t=${frame}, $P=$'+"%.8f" % total)
    return heatmap


anim = FuncAnimation(fig, update, frames=total_time, interval=total_time / anim_fps)

writer_video = animation.FFMpegWriter(fps=anim_fps)
anim.save(f'{timestamp}/{timestamp}.mp4', writer=writer_video)

plt.close('all')

print(str(tissue)+
      f'\n\nradius: {radius}\npol: {pol}, {tissue.pol2theta(pol)}\nori: {theta}\ntotal_time: {total_time}\nanim_fps: {anim_fps}', file=info)
info.close()
file.close()
sumj_data.close()
