#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

compacting_ratio = 1.0 # Lower is more compact but also more infinite loop
N = 4096
filename = f'init_{N}_med.dat'
draw_plot = True

# Set to None of False to not save

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

# maxd = np.sqrt(3/2. * N * compacting_ratio)

# Make a hex grid
#max_y = maxd - 1.0
#max_x = maxd - 1.0
#Nhor = sqrt(N * 2/sqrt(3))
#Nvert = sqrt(N * sqrt(3)/2)
h = int(round(np.sqrt(N * np.sqrt(0.75))))
#h = int((np.sqrt(1+np.sqrt(3)*8*N)-1)*.25)
pos = np.empty((2,N))
for i in range(N):
    pos[0,i] = (.5+1.5*(i//h))/np.sqrt(3)
for i in range(h):
    pos[1,i::2*h] = i+.5
for i in range(h,2*h):
    pos[1,i::2*h] = i-h

# how: height = h+.5 and width = ceil(N/h) * sqrt(.75)

maxd = np.sqrt(3/2. * N * compacting_ratio)
print(f"Number density = {N/(maxd*maxd)}")
print(f"Packing ratio  = {(np.pi/4)*N/(maxd*maxd)}")

# Expand and randomize
#sf = maxd/np.sqrt((h+.5)*np.ceil(N/float(h))*np.sqrt(0.75))
sf = maxd/(np.ceil(N/float(h))*np.sqrt(0.75))
pos *= sf # Space between points is now 1 (scale - 1)
angle  = np.random.uniform(0, 2*np.pi, pos.shape[1])
radius = np.random.uniform(0, (1-sf)/2., pos.shape[1])
pos[0] += radius * np.cos(angle)
pos[1] += radius * np.sin(angle)

# Compress
too_far_out = np.arange(pos.shape[1])[np.logical_or(pos[0] >= maxd-.5 , pos[1] >= maxd-.5)]
for idx in range(N):
    print(f"\rPlaced {idx} particles", end='')
    pos_ok = idx not in too_far_out
    mods_to_check = np.asarray(
            [[0,0],    [0, maxd],    [maxd, 0],        [0, -maxd],        [-maxd, 0],                  [maxd, maxd],                      [maxd, -maxd],                      [-maxd, maxd],                          [-maxd, -maxd]]) \
            [[True, pos[1,idx]<1, pos[0,idx]<1, pos[1,idx]>maxd-1, pos[0,idx]>maxd-1, pos[0,idx]<1 and pos[1,idx]<1, pos[0,idx]<1 and pos[1,idx]>maxd-1, pos[0,idx]>maxd-1 and pos[1,idx]<1, pos[0,idx]>maxd-1 and pos[1,idx]>maxd-1]]
    for mod in mods_to_check:
        dpos = pos - (pos[:,idx] + mod).reshape((2,1))
        overlapping = dpos[0]**2 + dpos[1]**2 <= 1
        overlapping[idx] = False
        if np.any(overlapping):
            pos_ok = False
            break
    if pos_ok:
        continue
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
print('')
# Plot
# https://stackoverflow.com/a/48174228/5618482  3 May 2020

#with open('init_{N}_med.dat') as f:
#    maxd, _ = np.genfromtxt(f, max_rows=1).tolist()
#    ip = np.genfromtxt(f)
#pos = ip[:,:2].T


if draw_plot:
    ax = plt.axes(aspect='equal')
    sc = scatter(*pos, ax, linewidth=0)
    scr = scatter(*(pos + [[0], [maxd]]), ax, linewidth=0)
    sct = scatter(*(pos + [[maxd], [0]]), ax, linewidth=0)
    sctr = scatter(*(pos + [[maxd], [maxd]]), ax, linewidth=0)
    plt.show()

ip = np.empty((3, pos.shape[1]))
ip[:2] = pos
ip[2] = np.random.rand(N) * (2*np.pi)

if filename:
    np.savetxt(filename, ip.T, header=f"{maxd} {maxd}", comments='')
