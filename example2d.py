# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:08:01 2014

@author: Tony Saad
"""
# !/usr/bin/env python
import os
import time
import spectra
import isoturb
import argparse
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from tkespec import compute_tke_spectrum2d


__author__ = 'Tony Saad'

pi = np.pi
energy_spectrums = ['cbc', 'vkp', 'kcm', 'homogeneous_isotropic']

if not os.path.isdir('./output'):
    os.mkdir('output')

# ----------------------------------------------------------------------------------------------
# __    __   ______   ________  _______         ______  __    __  _______
# |  \  |  \ /      \ |        \|       \       |      \|  \  |  \|       \ |  \  |  \|        \
# | $$  | $$|  $$$$$$\| $$$$$$$$| $$$$$$$\       \$$$$$$| $$\ | $$| $$$$$$
# | $$  | $$| $$___\$$| $$__    | $$__| $$        | $$  | $$$\| $$| $$__/ $$| $$  | $$   | $$
# | $$  | $$ \$$    \ | $$  \   | $$    $$        | $$  | $$$$\ $$| $$    $$| $$  | $$   | $$
# | $$  | $$ _\$$$$$$\| $$$$$   | $$$$$$$\        | $$  | $$\$$ $$| $$$$$$$ | $$  | $$   | $$
# | $$__/ $$|  \__| $$| $$_____ | $$  | $$       _| $$_ | $$ \$$$$| $$      | $$__/ $$   | $$
# \$$    $$ \$$    $$| $$     \| $$  | $$      |   $$ \| $$  \$$$| $$       \$$    $$   | $$
#  \$$$$$$   \$$$$$$  \$$$$$$$$ \$$   \$$       \$$$$$$ \$$   \$$ \$$        \$$$$$$     \$$
# ----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="This is the Utah Turbulence Generator.")

parser.add_argument(
    '-l',
    '--length',
    default=[9 * 2.0 * pi / 100.0] * 2,
    nargs=2,
    type=float,
    help="Domain size, lx ly (default is 0.18pi x 0.18pi)")
parser.add_argument(
    '-n',
    '--res',
    default=[64, 64],
    nargs=2,
    type=int,
    help="Grid resolution, nx ny (default is 64 x 64)")
parser.add_argument(
    '-m',
    '--modes',
    default=10000,
    nargs=1,
    type=int,
    help="Number of modes")
parser.add_argument(
    '-spec',
    '--spectrum',
    default=['cbc'],
    nargs=1,
    type=str,
    help="Select spectrum. Defaults to cbc. Other options include: vkp, kcm and homogeneous istropic.")

args = parser.parse_args()

# Default values for domain size in the x, y, and z directions. This value is typically
# based on the largest length scale that your data has. For the cbc data,
# the largest length scale corresponds to a wave number of 15, hence, the
# domain size is L = 2pi/15.
lx = args.length[0]
ly = args.length[1]

nx = args.res[0]
ny = args.res[1]

nmodes = args.modes
inputspec = args.spectrum[0]

# specify the spectrum name to append to all output filenames
fileappend = "{}_spec_{}x{}_{}x{}_{}_modes".format(
    inputspec, lx, ly, nx, ny, nmodes)

if inputspec not in energy_spectrums:
    raise Exception(
        "{} is not a supported spectrum. Supported spectrums: {}".format(
            inputspec, ", ".join(energy_spectrums)))

# now given a string name of the spectrum, find the corresponding function with the same name.
# use locals() because spectrum functions are defined in this module.
whichspec = getattr(spectra, '{}_spectrum'.format(inputspec))().evaluate

# enter the smallest wavenumber represented by this spectrum
wn1 = min(2.0 * pi / lx, 2.0 * pi / ly)

print(
    "\n---------------------SUMMARY OF USER INPUT---------------------\n"
    "Domain size: {} x {}\n"
    "Grid resolution: {} x {}\n"
    "Fourier accuracy (modes): {}\n"
    "Energy spectrum: {} (smallest wave number: {})\n"
    "---------------------------------------------------------------".format(
        lx, ly, nx, ny, nmodes, inputspec, wn1))


# input number of cells (cell centered control volumes). This will
# determine the maximum wave number that can be represented on this grid.
# see wnn below
dx = lx / nx
dy = ly / ny

t0 = time.time()
u, v = isoturb.generate_isotropic_turbulence_2d(
    lx, ly, nx, ny, nmodes, wn1, whichspec)
t1 = time.time()
elapsed_time = t1 - t0
print('it took me ', elapsed_time, 's to generate the isotropic turbulence.')


# compute mean velocities
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
np.savetxt('output/' + 'tkespec_' + fileappend +
           '.txt', np.transpose([wavenumbers, tkespec]))

# -------------------------------------------------------------
# compare spectra
# integral comparison:
# find index of nyquist limit
idx = (np.where(wavenumbers >= knyquist)[0][0]) - 2

# km0 = 2.0 * np.pi / lx
# km0 is the smallest wave number
km0 = wn1
# use a LOT of modes to compute the "exact" spectrum
# exactm = 10000
# dk0 = (knyquist - km0) / exactm
# exactRange = km0 + np.arange(0, exactm + 1) * dk0
dk = wavenumbers[1] - wavenumbers[0]
exactE = integrate.trapz(whichspec(wavenumbers[1:idx]), dx=dk)
print(f"exactE: {exactE}")
numE = integrate.trapz(tkespec[1:idx], dx=dk)
diff = np.abs((exactE - numE) / exactE)
integralE = diff * 100.0
print('Integral Error = ', integralE, '%')

# analyze how well we fit the input spectrum
# compute the RMS error committed by the generated spectrum
start_measure_index = 10
print(f"Start measure index: {start_measure_index}, "
      f"Wave number of the index: {wavenumbers[start_measure_index]}")
exact = whichspec(wavenumbers[start_measure_index:idx])
num = tkespec[start_measure_index:idx]
diff = np.abs((exact - num) / exact)
meanE = np.mean(diff)

print(f"Mean Error = {meanE * 100} %")
rmsE = np.sqrt(np.mean(diff * diff))
print('RMS Error = ', rmsE * 100, '%')


# create an array to save time and error values
array_toSave = np.zeros(4)
array_toSave[1] = integralE
array_toSave[0] = elapsed_time
array_toSave[2] = meanE * 100.0
array_toSave[3] = rmsE * 100.0

# save time and error values in a txt file
np.savetxt('output/' + 'time_error_' + fileappend + '.txt', array_toSave)
# np.savetxt('cpuTime_' + filespec + '_' + str(N) + '_' + str(nmodes) +
# '.txt',time_elapsed)

# -------------------------------------------------------------
# plt.figure()
# plt.imshow(u)
# plt.figure()
# plt.imshow(v)
# plt.show()

# plt.rc('text', usetex=True)
plt.rc("font", size=10, family='serif')

fig = plt.figure(figsize=(4.5, 3.8), dpi=200, constrained_layout=True)

wnn = np.arange(wn1, wavenumbers[-1] + 1000)

l1, = plt.loglog(wnn, whichspec(wnn), 'k-', label='input')
l2, = plt.loglog(wavenumbers[1:6], tkespec[1:6], 'bo--', markersize=3,
                 markerfacecolor='w', markevery=1, label='computed')
plt.loglog(wavenumbers[5:],
           tkespec[5:],
           'bo--',
           markersize=3,
           markerfacecolor='w',
           markevery=4,
           label='computed')
plt.axvline(x=knyquist, linestyle='--', color='black')
plt.xlabel(r'$\kappa$ (1/m)')
plt.ylabel(r'$E(\kappa)$ (m$^3$/s$^2$)')
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
            r_out = " ".join([str(100 * e) for e in u[r]])
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
