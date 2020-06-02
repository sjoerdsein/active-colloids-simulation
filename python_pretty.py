#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mcexercise as mce


viscosity = 10.0
propulsion_strength = 0.0005
nr_steps = 10000
skip_frames = 10
dt = 0.001

with open("init2d.dat") as file:
	box_size = np.genfromtxt(file, max_rows=1).tolist()
	initial_positions = np.genfromtxt(file)
	
result = mce.simulate(box_size, initial_positions, 
                      nr_steps,
                      skip_frames,
                      viscosity,
                      propulsion_strength,
                      dt)

print(np.shape(result[0,:,:2]))

fig = plt.figure("Simulation")
ax = fig.add_subplot(111,xlim=(0,box_size[0]), ylim = (0,box_size[1]))
pos = ax.scatter(result[0,:,0],result[0,:,1],c=np.linspace(0,256,num=256))
ppd=72./fig.dpi
trans = ax.transData.transform
s = ((trans((1,1))-trans((0,0)))*ppd)[1]
pos.set_sizes(np.full(result.shape[1], s**2))
ax.grid()
axbeta = plt.axes([0.15,0.03,0.65,0.03],facecolor = "lightgoldenrodyellow")
sbeta = Slider(axbeta, 't',0,int(nr_steps/skip_frames)-1, valstep = 1, valinit = 0)## slider

def update(val):
    t1 = int(sbeta.val)
    data = result[t1,:,:2]
    pos.set_offsets(data)

sbeta.on_changed(update)
plt.show()
