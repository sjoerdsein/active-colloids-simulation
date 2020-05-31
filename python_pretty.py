#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mcexercise as mce


nr_frames = 100

with open("init2d.dat") as file:
	box_size = np.genfromtxt(file, max_rows=1).tolist()
	initial_positions = np.genfromtxt(file)
	
result = mce.simulate(box_size, initial_positions, nr_frames)

#print(result[0,:,0])

fig = plt.figure("Simulation")
ax = fig.add_subplot(111,xlim=(0,box_size[0]), ylim = (0,box_size[1]))
pos = ax.scatter(result[0,:,0],result[0,:,1],c=np.linspace(0,256,num=256))
ax.grid()
axbeta = plt.axes([0.15,0.03,0.65,0.03],facecolor = "lightgoldenrodyellow")
sbeta = Slider(axbeta, 't',0,nr_frames, valstep = 1, valinit = 0)## slider

def update(val):
    t1 = int(sbeta.val)
    data = result[t1,:,:]
    pos.set_offsets(data)

sbeta.on_changed(update)
plt.show()
