#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
import mcexercise as mce

np.set_printoptions(precision=4, linewidth=332)

viscosity = 10.0
propulsion_strength = 0.001
repulsion_strength = 1
dt = 0.0001

density_scale_factor = 1.0
nr_densities = 1
frames_per_density = 1000
frame_interval = 10
init_equil_rounds = 3000
density_equil_rounds = 100

draw_MSD = False
MSD_filename = "MSD_active.svg"
draw_paths = False
render_animation = False
anim_filename = "jiggle_active.mp4"
draw_hist = False


nr_frames = nr_densities * frames_per_density
nr_steps = init_equil_rounds + nr_densities * (density_equil_rounds + nr_frames * frame_interval)

print(f"Péclet number: {propulsion_strength * 3 * np.pi * viscosity:.3f}")
print(f"Persistence time: {np.pi * viscosity:.3f}")
print(f"Total time: {nr_steps * dt:.3f}")
print(f"Performing {nr_steps} total simulation steps", end='')
if render_animation: print(f" and rendering {nr_frames} frames of animation", end='')

print("\nReading data")
with open("init_1024_med.dat") as file:
	box_size = np.genfromtxt(file, max_rows=1).tolist()
	initial_positions = np.genfromtxt(file)

f = 0.75 ## size factor
box_size = [p*f for p in box_size]
initial_positions[:,:2] *= f

print(f"Initial density is {initial_positions.shape[0] / (box_size[0] * box_size[1]):.3f} particles per unit area")
print(f"Final density is {initial_positions.shape[0] / (box_size[0] * box_size[1]) / (density_scale_factor ** (nr_densities * 2)):.3f} particles per unit area")

print("Running simulation")
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
print("Simulation done")

positions = result[...,:2]
densities = result[...,2]

friction_coefficient = 3 * np.pi * viscosity
diffusion_coefficient = 1.0/friction_coefficient

densities -= densities.min()
densities /= densities.max()

if draw_MSD:
    print("Drawing MSD")
    diff = positions - initial_positions[:,:2]
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
    if MSD_filename:
        plt.savefig(MSD_filename)
    plt.show()

if draw_paths:
    print("Drawing paths")
    reshaped = np.moveaxis(positions, -1, 0)
    ax = plt.axes(xlim=(0, box_size[0]), ylim=(0, box_size[1]), aspect=1)
    ax.plot(*reshaped, '.', ms=1)
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
    cmap = cm.get_cmap('viridis')

    def animate(i):
        scat.set_offsets(positions[i])
        scat.set_color(cmap(densities[i]))
        return scat,

    anim = animation.FuncAnimation(fig, animate, frames=nr_frames, blit=True, repeat=False)
    if anim_filename:
        anim.save(anim_filename, fps=30)
    
if draw_hist:
    plt.figure("Density histogram")
    #hist, xedges, yedges = np.histogram2d(densities.flatten(), bins=[
    plt.xlim(0,1)
    plt.hist(densities[-nr_frames//2:].flatten(), 200)
    plt.show()
