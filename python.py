#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import mcexercise as mce

np.set_printoptions(precision=4, linewidth=332)

nr_frames = 1000

draw_paths=True

render_animation = False
anim_filename = "jiggle.mp4"

with open("init2d.dat") as file:
	box_size = np.genfromtxt(file, max_rows=1).tolist()
	initial_positions = np.genfromtxt(file)


result = mce.simulate(box_size, initial_positions, nr_frames)

if draw_paths:
    reshaped = np.moveaxis(result, -1, 0)
    ax = plt.axes(xlim=(0, box_size[0]), ylim=(0, box_size[1]), aspect=1)
    ax.plot(*reshaped, '.', ms=1)
    plt.show()

if render_animation:
    # https://stackoverflow.com/a/48174228/5618482	3 May 2020
    fig = plt.figure(figsize=(4,4))
    ax = plt.axes(xlim=(0, box_size[0]), ylim=(0, box_size[1]), aspect=1)
    fig.canvas.draw()
    scat = ax.scatter(*initial_positions.T)
    ppd=72./fig.dpi
    trans = ax.transData.transform
    s = ((trans((1,1))-trans((0,0)))*ppd)[1]
    scat.set_sizes(np.full(result.shape[1], s**2))

    def animate(i):
        scat.set_offsets([*result[i]])
        return scat,

    anim = animation.FuncAnimation(fig, animate, frames=nr_frames, blit=True, repeat=False)
    anim.save("jiggle.mp4", fps=30)
