#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm
import mcexercise as mce


data_file = "init_512_med.dat"

viscosity = 5000.0
propulsion_strength = 0.01001
repulsion_strength = 1
dt = 0.0002

density_scale_factor = 1.05
nr_densities = 10
frames_per_density = 10
frame_interval = 100
init_equil_rounds = 3000
density_equil_rounds = 1000

rot_diff = 1/(np.pi * viscosity)
trans_diff = rot_diff/3
tau = 1/rot_diff
dt *= tau

nr_frames = nr_densities * frames_per_density
nr_steps = init_equil_rounds + nr_densities * (density_equil_rounds + nr_frames * frame_interval)

print(f"Péclet number: {propulsion_strength * tau:.3f}")
print(f"Persistence time: {tau:.3f}")
print(f"Total time: {nr_steps * dt:.3f}")
print(f"Delta time: {dt:.3f}")
print(f"Performing {init_equil_rounds + nr_densities * (density_equil_rounds + nr_frames * frame_interval)} total simulation steps and saving {nr_frames} frames")

print(f"Reading data from {data_file}")
with open(data_file) as file:
    box_size = np.genfromtxt(file, max_rows=1).tolist()
    initial_positions = np.genfromtxt(file)

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

densities = result[:,:,2]

densities = np.log(densities)
densities -= densities.min()
densities /= densities.max()

fig = plt.figure("Simulation")
ax = fig.add_subplot(121, xlim=(0,box_size[0]), ylim = (0,box_size[1]), aspect='equal')
ax.set_aspect('equal')
fig.canvas.draw()
pos = ax.scatter(result[0,:,0], result[0,:,1], c=densities[0])

ax2 = fig.add_subplot(122, xlim=(0,1), ylim=(0,100)) # Magic number 100
ax2.grid()
his = ax2.hist(densities[0],bins=12)

ppd = 72./fig.dpi
trans = ax.transData.transform
s = ((trans((1,1))-trans((0,0)))*ppd)[1]
pos.set_sizes(np.full(result.shape[1], s**2))
ax.grid()
axbeta = plt.axes([0.15,0.03,0.65,0.03], facecolor="lightgoldenrodyellow")
sbeta = Slider(axbeta, 't', 0, nr_frames-1, valstep=1, valinit=0, valfmt="%0.0f") # slider

cmap = cm.get_cmap('viridis')

def update(val):
    t1 = int(val)
    data = result[t1,:,:2]
    pos.set_offsets(data)
    pos.set_color(cmap(densities[t1]))
    ax.set_xlim(0, box_size[0] * density_scale_factor ** (t1 // frames_per_density))
    ax.set_ylim(0, box_size[0] * density_scale_factor ** (t1 // frames_per_density))
    ppd=72./fig.dpi
    trans = ax.transData.transform
    s = ((trans((1,1))-trans((0,0)))*ppd)[1]
    pos.set_sizes(np.full(result.shape[1], s**2))

    ax2.clear()
    ax2.grid()
    ax2.hist(densities[t1],bins=12)
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,100)

sbeta.on_changed(update)
plt.show()
