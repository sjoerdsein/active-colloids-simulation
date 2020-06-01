#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import mcexercise as mce

np.set_printoptions(precision=4, linewidth=332)

viscosity = 10.0
propulsion_strength = 0.0005
nr_steps = 10000
skip_frames = 10
dt = 0.001


draw_MSD = False
MSD_filename = "MSD_active.svg"
draw_paths = False
render_animation = False
anim_filename = "jiggle_active.mp4"

nr_frames = nr_steps // skip_frames
print(f"Performing {nr_steps} simulation steps", end='')
if render_animation: print(f" and rendering {nr_frames} frames of animation", end='')

print("\nReading data")
with open("init2d.dat") as file:
	box_size = np.genfromtxt(file, max_rows=1).tolist()
	initial_positions = np.genfromtxt(file)

print("Runnung simulation")
result = mce.simulate(box_size,
                      initial_positions,
                      nr_steps,
                      skip_frames,
                      viscosity,
                      propulsion_strength,
                      dt)
print("Simulation done")

friction_coefficient = 3 * np.pi * viscosity
diffusion_coefficient = 1.0/friction_coefficient

if draw_MSD:
    print("Drawing MSD")
    diff = result - initial_positions
    plt.grid(True)
    # plt.plot(np.arange(nr_frames) * dt * 2 * 2 * diffusion_coefficient * skip_frames, 'r--', label=r"Expected $\langle r^2 \rangle$")  # For an open space and no interactions
    plt.plot(np.mean(diff[...,0]**2 + diff[...,1]**2, axis=1), 'k', label=r"Actual $\langle r^2 \rangle$")
    plt.xlim(0,nr_frames)
    plt.ylim(0,None)
    plt.xlabel("Step nr")
    plt.ylabel(r"$\langle r^2\rangle$")
    # plt.axhline((box_size[0]**2 + box_size[1]**2)/6, c='k')  # Expected <r^2> when all particles are randomly distributed inside a box
    plt.legend()
    plt.title(fr"Diffusion coefficient $D={diffusion_coefficient:.2}$")
    plt.savefig(MSD_filename)
    plt.show()


if draw_paths:
    print("Drawing paths")
    reshaped = np.moveaxis(result, -1, 0)
    ax = plt.axes(xlim=(0, box_size[0]), ylim=(0, box_size[1]), aspect=1)
    ax.plot(*reshaped[...,:2], '.', ms=1)
    plt.show()

if render_animation:
    print("Rendering animation")
    # https://stackoverflow.com/a/48174228/5618482	3 May 2020
    fig = plt.figure(figsize=(4,4))
    ax = plt.axes(xlim=(0, box_size[0]), ylim=(0, box_size[1]), aspect=1)
    fig.canvas.draw()
    scat = ax.scatter(*initial_positions[...,:2].T)
    ppd=72./fig.dpi
    trans = ax.transData.transform
    s = ((trans((1,1))-trans((0,0)))*ppd)[1]
    scat.set_sizes(np.full(result.shape[1], s**2))

    def animate(i):
        scat.set_offsets([*result[i,:,:2]])
        return scat,

    anim = animation.FuncAnimation(fig, animate, frames=nr_frames, blit=True, repeat=False)
    anim.save(anim_filename, fps=30)
