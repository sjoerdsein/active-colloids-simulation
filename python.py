#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
import mcexercise as mce
from scipy import odr

np.set_printoptions(precision=4, linewidth=332)

init_file ="init_512_dense.dat"

viscosity = 5000.0
propulsion_strength = 0.00 # v
dt = 0.0001

density_scale_factor = 1.0
nr_densities = 1
frames_per_density = 1#000
frame_interval = 200
init_equil_rounds = 50000#0
#init_equil_time = 100
density_equil_rounds = 0

draw_MSD = False
MSD_filename = "MSD_active.svg"
draw_paths = False
render_animation = False
anim_filename = "jiggle_active.mp4"
draw_hist = False
draw_frame = True


rot_diff = 1./(np.pi * viscosity)
trans_diff = rot_diff/3.
tau = 1./rot_diff
dt *= tau
peclet = propulsion_strength * tau

#init_equil_rounds = int(init_equil_time * tau)

#init_equil_rounds = int(init_equil_time * tau)

nr_frames = nr_densities * frames_per_density
nr_steps = init_equil_rounds + nr_densities * (density_equil_rounds + nr_frames * frame_interval)

print(f"Péclet number: {peclet:.3f}")
print(f"Persistence time: {tau:.3f}")
print(f"Total time: {nr_steps * dt:.3f}")
print(f"Delta time: {dt:.3f}")
print(f"Performing {init_equil_rounds + nr_densities * (density_equil_rounds + nr_frames * frame_interval)} total simulation steps and saving {nr_frames} frames")

hist_filename = f"hist_512_dense_{peclet:.1f}.pdf"

print("Reading data")
with open(init_file) as file:
	box_size = np.genfromtxt(file, max_rows=1).tolist()
	initial_positions = np.genfromtxt(file)

print(f"Density is {initial_positions.shape[0] / (box_size[0] * box_size[1]):.3f} particles per unit area")
print(f"Packing fraction is {initial_positions.shape[0] / (box_size[0] * box_size[1]) * np.pi/4.0:.3f}")

print("Running simulation")
result = mce.simulate(box_size,
                      initial_positions,
                      viscosity,
                      propulsion_strength,
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


if draw_MSD:
    print("Drawing MSD")
    diff = positions - initial_positions[:,:2]
    plt.grid(True)
    plt.plot(np.mean(diff[...,0]**2 + diff[...,1]**2, axis=1), 'k', label=r"Actual $\langle r^2 \rangle$")
    plt.xlim(0,nr_frames)
    plt.ylim(0,None)
    plt.xlabel("Step nr")
    plt.ylabel(r"$\langle r^2\rangle$")
    plt.legend()
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
    ppd = 72./fig.dpi
    trans = ax.transData.transform
    s = ((trans((1,1))-trans((0,0)))*ppd)[1]
    scat.set_sizes(np.full(result.shape[1], s**2))
    cmap = cm.get_cmap('viridis')

    densities01  = densities - densities.min()
    densities01 /= densities01.max()

    def animate(i):
        scat.set_offsets(positions[i])
        scat.set_color(cmap(densities01[i]))
        return scat,

    anim = animation.FuncAnimation(fig, animate, frames=nr_frames, blit=True, repeat=False)
    if anim_filename:
        anim.save(anim_filename, fps=30)

if draw_hist:
    print("Drawing histogram")
    plt.rcParams['text.latex.preamble']=[r"\usepackage{fourier}\usepackage{ebgaramond}\DeclareMathSizes{10}{9}{7}{5}"]
    plt.rcParams.update({
        'text.usetex' : True,
        'font.size' : 10,
        'font.family' : 'ebgaramond' })
    plt.figure("Density histogram", figsize=(1.75,1.25), constrained_layout=True)
    #hist, xedges, yedges = np.histogram2d(densities.flatten(), bins=[
    #plt.xlim(0,1)
    heights, edges, _ = plt.hist(densities.flatten(), 200, density=True, color='#d6d6d9')
    edges += (edges[1] - edges[0])/2.0

    def bell_curve(x, mean, stddev):
        return np.exp(-(((x-mean)/stddev)**2)/2)/(np.sqrt(2*np.pi)*stddev)
    def two_bell_curves(B, x):
        return bell_curve(x, B[0], B[1]) * B[4] + bell_curve(x, B[2], B[3]) * (1.0-B[4])

    odr_model  = odr.Model(two_bell_curves)
    odr_data   = odr.Data(edges[:-1], heights)
    odr_odr    = odr.ODR(odr_data, odr_model, beta0=[0.7, 0.2, 1.1, 0.15, 0.5])
    odr_output = odr_odr.run()

    b = odr_output.beta
    print(f"Curve 1: mean={b[0]:.3f}, width={b[1]:.3f}, share={    b[4]:.3f}")
    print(f"Curve 2: mean={b[2]:.3f}, width={b[3]:.3f}, share={1.0-b[4]:.3f}")

    print(f"{peclet:.1f}")
    print(f"{b[0]:.3f} {b[1]:.3f} {    b[4]:.3f}")
    print(f"{b[2]:.3f} {b[3]:.3f} {1.0-b[4]:.3f}")

    plt.plot(odr_data.x, odr_model.fcn(odr_output.beta, odr_data.x), c='xkcd:bright blue')
    plt.plot(odr_data.x, bell_curve(odr_data.x, *b[0:2])*     b[4] , c='tab:red')
    plt.plot(odr_data.x, bell_curve(odr_data.x, *b[2:4])*(1.0-b[4]), c='tab:red')

    plt.xlim(0,None)
    ymax = plt.gca().get_ylim()[1]

    plt.axvline(b[0], ymax=(     b[4] /(b[1]*np.sqrt(2*np.pi)))/ymax, c='tab:red', ls='--')
    plt.axvline(b[2], ymax=((1.0-b[4])/(b[3]*np.sqrt(2*np.pi)))/ymax, c='tab:red', ls='--')
    plt.xlabel(r"Local density $\rho=\frac{1}{A_\mathrm{cell}}$")
    plt.ylabel(r"$P(\rho)$")
    #plt.tight_layout()

    if hist_filename:
        plt.savefig(hist_filename)
    plt.show()

if draw_frame:
    plt.rcParams['text.latex.preamble']=[r"""
        \usepackage{mathtools}
        \usepackage{fourier}
        \usepackage{ebgaramond}
        \DeclareMathSizes{10}{9}{7}{5}
    """]
    plt.rcParams.update({
        'text.usetex' : True,
        'font.size' : 10,
        'font.family' : 'ebgaramond' })
    fig = plt.figure(figsize=(1.75,1.75), constrained_layout=True)
    ax = plt.axes(xlim=(0, box_size[0]), ylim=(0, box_size[1]), aspect=1)
    fig.canvas.draw()
    scat = ax.scatter(*positions.T)
    ppd = 72./fig.dpi
    trans = ax.transData.transform
    s = ((trans((1,1))-trans((0,0)))*ppd)[1]
    scat.set_sizes(np.full(result.shape[1], s**2))
    cmap = cm.get_cmap('viridis')

    densities01  = densities - densities.min()
    densities01 /= densities01.max()

    scat.set_color(cmap(densities01[0]))
    plt.savefig(f"frame_{peclet:.1f}.pdf")
