#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mcexercise as mce

## ___ this is all simply copied from python_pretty.py

viscosity = 10.0
propulsion_strength = 0.003
repulsion_strength = 1.0
dt = 0.001

density_scale_factor = 1.05
nr_densities = 10
frames_per_density = 10
frame_interval = 10
init_equil_rounds = 300
density_equil_rounds = 100

with open("init2d.dat") as file:
	box_size = np.genfromtxt(file, max_rows=1).tolist()
	initial_positions = np.genfromtxt(file)

f = 0.75 ## size factor
box_size = [r*f for r in box_size]
initial_positions[:,:2] *= f


result = mce.simulate(box_size,
                      initial_positions,
                      viscosity,
                      propulsion_strength,
                      repulsion_strength,
                      dt,
                      density_scale_factor,
                      nr_densities,
                      frames_per_density,
                      frame_interval,
                      init_equil_rounds,
                      density_equil_rounds)

densities = result[:,:,2]

densities -= densities.min()
densities /= densities.max()
## ___

nr_frames = nr_densities * frames_per_density
nr_steps = nr_frames * frame_interval

no_mom = 7 ## no. of moments to get
snippits = densities[0::int(nr_steps/(frame_interval*(no_mom-1)))]
#print(snippits[0])

#tp0 = densities[0]
tp0 = snippits[0]
tp = snippits.ravel()
tp = np.append(tp,densities[-1])

t = np.zeros_like(tp)


moments = np.zeros(no_mom)
for i in range(0,no_mom):
    moments[i] = dt*i*nr_steps/(no_mom-1)
    t[i*nr_frames:(i+1)*nr_frames] = np.full(np.shape(t[i*nr_frames:(i+1)*nr_frames]),moments[i])

print(t[0::256])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
t0 = np.zeros_like(tp0)

ybins = 20

hist0,xedges0,yedges0 = np.histogram2d(t,tp, bins = [int(no_mom),ybins],range=[[0,int(nr_steps*dt)],[0,1]])


# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges0[:-1], yedges0[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = 0.07 * np.ones_like(zpos)*int(nr_steps*dt)*4/no_mom
dy = 0.7 * np.ones_like(zpos)/ybins
dz = hist0.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax.set_xlabel("t")
ax.set_ylabel("Local density")
ax.set_zlabel("# of particles")
ax.grid()
plt.show()
