#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm
import mcexercise as mce


viscosity = 10.0
propulsion_strength = 0.00015
nr_steps = 30000
skip_frames = 10
dt = 0.001

with open("init2d.dat") as file:
	box_size = np.genfromtxt(file, max_rows=1).tolist()
	initial_positions = np.genfromtxt(file)

f = 0.80 ## size factor
box_size = [r*f for r in box_size]
initial_positions[:,:2] *= f


result = mce.simulate(box_size, initial_positions, 
                      nr_steps,
                      skip_frames,
                      viscosity,
                      propulsion_strength,
                      dt)

densities = result[:,:,2]

densities -= densities.min()
densities /= densities.max()


fig = plt.figure("Simulation")
ax = fig.add_subplot(121,xlim=(0,box_size[0]), ylim = (0,box_size[1]))
axis1 = plt.gca()
axis1.set_aspect('equal')
pos = ax.scatter(result[0,:,0],result[0,:,1],c=densities[0])

ax2 = fig.add_subplot(122,xlim=(0,1),ylim=(0,100))
axis2 = plt.gca()
his = ax2.hist(densities[0],bins=12)
ax2.grid()

ppd=72./fig.dpi
trans = ax.transData.transform
s = ((trans((1,1))-trans((0,0)))*ppd)[1]
pos.set_sizes(np.full(result.shape[1], s**2))
ax.grid()
axbeta = plt.axes([0.15,0.03,0.65,0.03],facecolor = "lightgoldenrodyellow")
sbeta = Slider(axbeta, 't',0,int(nr_steps/skip_frames)-1, valstep = 1, valinit = 0)## slider

cmap = cm.get_cmap('viridis')

def update(val):
    t1 = int(sbeta.val)
    data = result[t1,:,:2]
    pos.set_offsets(data)
    pos.set_color(cmap(densities[t1]))
    
    axis2.clear()
    ax2.hist(densities[t1],bins=12)
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,100)
    ax2.grid()
sbeta.on_changed(update)
plt.show()

