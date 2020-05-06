#!/usr/bin/env python
import numpy as np
import mcexercise as mce

np.set_printoptions(precision=4, linewidth=332)

nr_frames = 1000

with open("init2d.dat") as file:
	box_size = np.genfromtxt(file, max_rows=1).tolist()
	initial_positions = np.genfromtxt(file)


result = mce.simulate(box_size, initial_positions, nr_frames)
print(result.shape)

