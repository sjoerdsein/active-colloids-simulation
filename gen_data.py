#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

compacting_ratio = 0.7071 # sqrt(0.5) (0.7071) yields scale factor of sqrt(3) (1.732)
# Lower is more compact but also more infinite loop

draw_plot = True

filename = "init2d.dat"
# Set to None of False to not save

# Make a hex grid
pos = np.empty((2,256))
for i in range(256):
	pos[0,i] = (.5+1.5*(i//16))/np.sqrt(3)
for i in range(16):
	pos[1,i::32] = i+.5
for i in range(16,32):
	pos[1,i::32] = i-16

# Expand and randomize
pos *= 2 # Space between points is now 1
angle  = np.random.uniform(0, 2*np.pi, pos.shape[1])
radius = np.random.uniform(0, .5, pos.shape[1])
pos[0] += radius * np.cos(angle)
pos[1] += radius * np.sin(angle)

# Compress
maxd = 48/np.sqrt(3) * compacting_ratio
too_far_out = np.arange(pos.shape[1])[np.logical_or(pos[0] >= maxd-.5 , pos[1] >= maxd-.5)]
for idx in too_far_out:
	pos_ok = False
	while not pos_ok:
		newpos = np.random.uniform(0, maxd, 2)
		mods_to_check = np.asarray(
		        [[0,0], [0, maxd],   [maxd, 0],   [0, -maxd],       [-maxd, 0],       [maxd, maxd],                [maxd, -maxd],                    [-maxd, maxd],                    [-maxd, -maxd]]) \
		        [[True, newpos[1]<1, newpos[0]<1, newpos[1]>maxd-1, newpos[0]>maxd-1, newpos[0]<1 and newpos[1]<1, newpos[0]<1 and newpos[1]>maxd-1, newpos[0]>maxd-1 and newpos[1]<1, newpos[0]>maxd-1 and newpos[1]>maxd-1]]
		for mod in mods_to_check:
			dpos = pos - (newpos + mod).reshape((2,1))
			if np.any(dpos[0]**2 + dpos[1]**2 <= 1):
			    break

		else: # Belongs to for-loop
			pos_ok = True

	pos[:,idx] = newpos

# Plot
# https://stackoverflow.com/a/48174228/5618482	3 May 2020
class scatter():
	def __init__(self,x,y,ax,size=1,**kwargs):
		self.n = len(x)
		self.ax = ax
		self.ax.figure.canvas.draw()
		self.size_data=size
		self.size = size
		self.sc = ax.scatter(x,y,s=self.size,**kwargs)
		self._resize()
		self.cid = ax.figure.canvas.mpl_connect('draw_event', self._resize)

	def _resize(self,event=None):
		ppd=72./self.ax.figure.dpi
		trans = self.ax.transData.transform
		s =  ((trans((1,self.size_data))-trans((0,0)))*ppd)[1]
		if s != self.size:
			self.sc.set_sizes(s**2*np.ones(self.n))
			self.size = s
			self._redraw_later()

	def _redraw_later(self):
		self.timer = self.ax.figure.canvas.new_timer(interval=10)
		self.timer.single_shot = True
		self.timer.add_callback(lambda : self.ax.figure.canvas.draw_idle())
		self.timer.start()

if draw_plot:
	sc = scatter(*pos, plt.gca(), linewidth=0)
	scr = scatter(*(pos + [[0], [maxd]]), plt.gca(), linewidth=0)
	sct = scatter(*(pos + [[maxd], [0]]), plt.gca(), linewidth=0)
	sctr = scatter(*(pos + [[maxd], [maxd]]), plt.gca(), linewidth=0)
	plt.gca().set_aspect('equal')
	plt.show()

if filename:
    np.savetxt(filename, pos.T, header=f"{maxd} {maxd}", comments='')
