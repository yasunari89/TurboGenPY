# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:08:01 2014

@author: Tony Saad
"""
# !/usr/bin/env python
from scipy import interpolate
from scipy import integrate
import numpy as np
from numpy import pi
import time
import scipy.io
from spectra import homogeneous_isotropic_spectrum
from tkespec import compute_tke_spectrum2d
import isoturb
import isoturbo
from fileformats import FileFormats
import isoio
import cudaturbo

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

#plt.interactive(True)

import spectra

# ----------------------------------------------------------------------------------------------
# __    __   ______   ________  _______         ______  __    __  _______   __    __  ________
# |  \  |  \ /      \ |        \|       \       |      \|  \  |  \|       \ |  \  |  \|        \
# | $$  | $$|  $$$$$$\| $$$$$$$$| $$$$$$$\       \$$$$$$| $$\ | $$| $$$$$$$\| $$  | $$ \$$$$$$$$
# | $$  | $$| $$___\$$| $$__    | $$__| $$        | $$  | $$$\| $$| $$__/ $$| $$  | $$   | $$
# | $$  | $$ \$$    \ | $$  \   | $$    $$        | $$  | $$$$\ $$| $$    $$| $$  | $$   | $$
# | $$  | $$ _\$$$$$$\| $$$$$   | $$$$$$$\        | $$  | $$\$$ $$| $$$$$$$ | $$  | $$   | $$
# | $$__/ $$|  \__| $$| $$_____ | $$  | $$       _| $$_ | $$ \$$$$| $$      | $$__/ $$   | $$
# \$$    $$ \$$    $$| $$     \| $$  | $$      |   $$ \| $$  \$$$| $$       \$$    $$   | $$
#  \$$$$$$   \$$$$$$  \$$$$$$$$ \$$   \$$       \$$$$$$ \$$   \$$ \$$        \$$$$$$     \$$
# ----------------------------------------------------------------------------------------------


import argparse
__author__ = 'Tony Saad'
parser = argparse.ArgumentParser(description='This is the Utah Turbulence Generator.')
parser.add_argument('-l' , '--length', help='Domain size, lx ly lz',required=False, nargs='+', type=float)
parser.add_argument('-n' , '--res'  , help='Grid resolution, nx ny nz',required=False, nargs='+', type=int)
parser.add_argument('-m' , '--modes' , help='Number of modes', required=False,type=int)
parser.add_argument('-gpu', '--cuda', help='Use a GPU if availalbe', required = False, action='store_true')
parser.add_argument('-mp' , '--multiprocessor',help='Use the multiprocessing package', required = False,nargs='+', type=int)
parser.add_argument('-o'  , '--output', help='Write data to disk', required = False,action='store_true')
parser.add_argument('-spec', '--spectrum', help='Select spectrum. Defaults to cbc. Other options include: vkp, and kcm.', required = False, type=str)
args = parser.parse_args()

# parse grid resolution (nx, ny, nz).
nx = 64
ny = 64

if args.res:
	N = args.res
	if len(N) == 1:
		nx = ny = N[0]
	else:
		nx = N[0]
		ny = N[1]

# Default values for domain size in the x, y, and z directions. This value is typically
# based on the largest length scale that your data has. For the cbc data,
# the largest length scale corresponds to a wave number of 15, hence, the
# domain size is L = 2pi/15.
lx = 9 * 2.0 * pi / 100.0
ly = 9 * 2.0 * pi / 100.0

# parse domain length, lx, ly, and lz
L = args.length
if L:
	if len(L) == 1:
		lx = ly = L[0]
	elif len(L) == 2:
		lx = L[0]
		ly = L[1]

# parse number of modes
nmodes = 10000
m = args.modes
if m:
	nmodes = int(m)
	print(m)


# specify which spectrum you want to use. Options are: cbc_spec, vkp_spec, and power_spec
inputspec = 'cbc'
if args.spectrum:
	inputspec = args.spectrum

# specify the spectrum name to append to all output filenames
fileappend = inputspec + '_' + str(nx) + '.' + str(ny) + '_' + str(nmodes) + '_modes'

print('input spec', inputspec)
if inputspec != 'cbc' and inputspec != 'vkp' and inputspec != 'kcm' and inputspec != 'homogeneous_isotropic':
	print('Error: ', inputspec, ' is not a supported spectrum. Supported spectra are: cbc, vkp, homogeneous isotropic and power. Please revise your input.')
	exit()
inputspec += '_spectrum'
# now given a string name of the spectrum, find the corresponding function with the same name. use locals() because spectrum functions are defined in this module.
# whichspec = locals()[inputspec]
# whichspec = spectra.cbc_spectrum().evaluate
whichspec = getattr(spectra, inputspec)().evaluate

# write to file
enableIO = False  # enable writing to file
io = args.output
if io:
	enableIO = io
fileformat = FileFormats.FLAT  # Specify the file format supported formats are: FLAT, IJK, XYZ

# save the velocity field as a matlab matrix (.mat)
savemat = False

# compute the mean of the fluctuations for verification purposes
computeMean = True

# check the divergence of the generated velocity field
checkdivergence = False

# enter the smallest wavenumber represented by this spectrum
wn1 = min(2.0*pi/lx, 2.0*pi/ly)
# wn1 = 15  # determined here from cbc spectrum properties

# summarize user input
print('-----------------------------------')
print('SUMMARY OF USER INPUT:')
print('Domain size:', lx, ly)
print('Grid resolution:', nx, ny)
print('Fourier accuracy (modes):', nmodes)
print(f"Smallest wave number: {wn1}")


# ------------------------------------------------------------------------------
# END USER INPUT
# ------------------------------------------------------------------------------

# input number of cells (cell centered control volumes). This will
# determine the maximum wave number that can be represented on this grid.
# see wnn below
dx = lx / nx
dy = ly / ny

t0 = time.time()
u, v = isoturb.generate_isotropic_turbulence_2d(lx, ly, nx, ny, nmodes, wn1, whichspec)
t1 = time.time()
elapsed_time = t1 - t0
print('it took me ', elapsed_time, 's to generate the isotropic turbulence.')


# compute mean velocities
if computeMean:
	umean = np.mean(u)
	vmean = np.mean(v)

	print('mean u = ', umean)
	print('mean v = ', vmean)


	ufluc = umean - u
	vfluc = vmean - v


	print('mean u fluct = ', np.mean(ufluc))
	print('mean v fluct = ', np.mean(vfluc))


	ufrms = np.mean(ufluc * ufluc)
	vfrms = np.mean(vfluc * vfluc)


	print('u fluc rms = ', np.sqrt(ufrms))
	print('v fluc rms = ', np.sqrt(vfrms))

# verify that the generated velocities fit the spectrum
knyquist, wavenumbers, tkespec = compute_tke_spectrum2d(u, v, lx, ly, False)
print(f"Knyquist: {knyquist}")
print(f"Wave numbers shape: {wavenumbers.shape}")
print(f"Discrete sperctrum shape: {tkespec.shape}")
# save the generated spectrum to a text file for later post processing
np.savetxt('output/' + 'tkespec_' + fileappend + '.txt', np.transpose([wavenumbers, tkespec]))

# -------------------------------------------------------------
# compare spectra
# integral comparison:
# find index of nyquist limit
idx = (np.where(wavenumbers >= knyquist)[0][0]) - 2

# km0 = 2.0 * np.pi / lx
# km0 is the smallest wave number
km0 = wn1
# use a LOT of modes to compute the "exact" spectrum
#exactm = 10000
#dk0 = (knyquist - km0) / exactm
#exactRange = km0 + np.arange(0, exactm + 1) * dk0
dk = wavenumbers[1] - wavenumbers[0]
exactE = integrate.trapz(whichspec(wavenumbers[1:idx]), dx=dk)
print(f"exactE: {exactE}")
numE = integrate.trapz(tkespec[1:idx], dx=dk)
diff = np.abs((exactE - numE)/exactE)
integralE = diff*100.0
print('Integral Error = ', integralE, '%')

# analyze how well we fit the input spectrum
# compute the RMS error committed by the generated spectrum
start_measure_index = 10
print(f"Start measure index: {start_measure_index}, "\
		f"Wave number of the index: {wavenumbers[start_measure_index]}")
exact = whichspec(wavenumbers[start_measure_index:idx])
num = tkespec[start_measure_index:idx]
diff = np.abs((exact - num) / exact)
meanE = np.mean(diff)

print(f"Mean Error = {meanE * 100} %")
rmsE = np.sqrt(np.mean(diff * diff))
print('RMS Error = ', rmsE * 100, '%')


#create an array to save time and error values
array_toSave = np.zeros(4)
array_toSave[1] = integralE
array_toSave[0] = elapsed_time
array_toSave[2] = meanE*100.0
array_toSave[3] = rmsE*100.0

# save time and error values in a txt file
np.savetxt('output/' + 'time_error_' + fileappend + '.txt', array_toSave)
#np.savetxt('cpuTime_' + filespec + '_' + str(N) + '_' + str(nmodes) + '.txt',time_elapsed)

# -------------------------------------------------------------
# plt.figure()
# plt.imshow(u)
# plt.figure()
# plt.imshow(v)
# plt.show()

# plt.rc('text', usetex=True)
plt.rc("font", size=10, family='serif')

fig = plt.figure(figsize=(4.5, 3.8), dpi=200, constrained_layout=True)

wnn = np.arange(wn1, wavenumbers[-1]+1000)

l1, = plt.loglog(wnn, whichspec(wnn), 'k-', label='input')
l2, = plt.loglog(wavenumbers[1:6], tkespec[1:6], 'bo--', markersize=3, markerfacecolor='w', markevery=1, label='computed')
plt.loglog(wavenumbers[5:], tkespec[5:], 'bo--', markersize=3, markerfacecolor='w', markevery=4, label='computed')
plt.axvline(x=knyquist, linestyle='--', color='black')
plt.xlabel('$\kappa$ (1/m)')
plt.ylabel('$E(\kappa)$ (m$^3$/s$^2$)')
plt.grid()
# plt.gcf().tight_layout()
if nx == ny:
	plt.title(str(nx) + '$^2$')
else:
	plt.title(str(nx) + 'x' + str(ny))
plt.legend(handles=[l1, l2], loc=1)
# fig.savefig('tkespec_' + filespec + '_' + str(N) + '.pdf')
fig.savefig('output/' + 'tkespec_' + fileappend + '.pdf')
# plt.show()

# add output of velocity u and v
velocity_kinds = (
	(u, 'velocity_u_{}'.format(fileappend)),
	(v, 'velocity_v_{}'.format(fileappend))
)
for v_kind in velocity_kinds:
	velocity = v_kind[0]
	v_file_name = 'output/' + v_kind[1] + '.txt'
	with open(v_file_name, 'w') as f:
		row_size, col_size = velocity.shape
		for r in range(row_size):
			# m/s --> cm/s
			r_out = " ".join([str(100*e) for e in u[r]])
			f.write("{}\n".format(r_out))

for v_kind in velocity_kinds:
	velocity = v_kind[0]
	v_file_name = 'output/' + v_kind[1] + '.pdf'
	fig = plt.figure(dpi=200, constrained_layout=True)
	plt.cla()
	plt.imshow(velocity)
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.colorbar()
	plt.savefig(v_file_name, bbox_inches='tight')