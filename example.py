# import colorcet as cc
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
from obspy.core import UTCDateTime
import obspy.geodetics.base as obs
from obspy import read

from waveform_collection import gather_waveforms
from tcm.algorithms import tcm

import os as os
os.chdir('/Users/jwbishop/Desktop/Matt_TCM_Code/tcm_code_20211103')

# Read example data
import scipy.io
mat = scipy.io.loadmat('dta.mat')['dta']
# Reading the metadata is a mess - fs = 50 Hz.
# dum = scipy.io.loadmat('dum.mat')['dum'][0][0]
# dum.tolist()

nchan = np.shape(mat)[0]
fs = 50

# Filtering
fmin = 10.0  # [Hz]
fmax = 20.0  # [Hz]

winlen = 4  # [s]
WIND = WINLEN * fs
WINOVER = 0.10
WINDOW = WIND/2
nfft = np.power(2, int(np.ceil(np.log2(WIND))))
fss = fs

# Data Mapping
I = mat[0, :]
E = mat[3, :]
N = mat[2, :]
Z = mat[1, :]

# Matt's `wind`
winlensamp = int(winlen*fs)
# 90% overlap
winover = 0.90
window = winlensamp/2
sampinc = int((1 - winover) * winlensamp) + 1
npts = len(I) - winlensamp
its = np.arange(0, npts, sampinc)
nits = len(its)

from scipy.signal import csd
from scipy.fft import rfftfreq
freq = rfftfreq(nfft, 1/fs)
# freq = freq[np.where((freq >= FMIN) & (freq <= FMAX))]

Cxy2zi = np.empty((nits, len(freq)), dtype=np.complex128)
Cxy2ii = np.empty((nits, len(freq)), dtype=np.complex128)
Cxy2zz = np.empty((nits, len(freq)), dtype=np.complex128)
# print('Running tcm for %d windows' % nits)
# Compute the vertical coherence
for jj in range(0, nits):
    # Get time from middle of window, except for the end.
    ptr = [int(its[jj]), int(its[jj] + winlensamp)]
    # try:
    #     t[jj] = tvec[ptr[0]+int(winlensamp/2)]
    # except:
    #     t[jj] = np.nanmax(t, axis=0)

    f, Cxy2zi[jj, :] = csd(Z[ptr[0]:ptr[1]], I[ptr[0]:ptr[1]], fs=fs, window='hann', nperseg=None, noverlap=None, nfft=nfft)
    f, Cxy2ii[jj, :] = csd(I[ptr[0]:ptr[1]], I[ptr[0]:ptr[1]], fs=fs, window='hann', nperseg=None, noverlap=None, nfft=nfft)
    f, Cxy2zz[jj, :] = csd(Z[ptr[0]:ptr[1]], Z[ptr[0]:ptr[1]], fs=fs, window='hann', nperseg=None, noverlap=None, nfft=nfft)

Cxy2 = np.real(np.multiply(Cxy2zi, np.conjugate(Cxy2zi))/np.multiply(Cxy2ii, Cxy2zz))
Cxy2g = np.abs(Cxy2zi)/np.real(Cxy2ii)

# Time Vector
tvecC = its/fs
# Add half the window length [sec]
tvecC_z = tvecC + winlen/2

# azimuths to scan over
azvect = np.array(np.arange(-179, 181, 1))

# Get the closest frequency points to the preferred ones
fmin_ind = np.argmin(np.abs(fmin - f))
fmax_ind = np.argmin(np.abs(fmax - f))

Cey2h2p = np.empty((nits, len(freq)), dtype=np.complex128)
Cny2h2p = np.empty((nits, len(freq)), dtype=np.complex128)
Cee2h2p = np.empty((nits, len(freq)), dtype=np.complex128)
Cnn2h2p = np.empty((nits, len(freq)), dtype=np.complex128)
Cne2h2p = np.empty((nits, len(freq)), dtype=np.complex128)
Cyy2h2p = np.empty((nits, len(freq)), dtype=np.complex128)
# Loop over windows
for jj in range(0, nits):
    ptr = [int(its[jj]), int(its[jj] + winlensamp)]
    # We only need 7 cross spectra for the angle sum
    f, Cey2h2p[jj, :] = csd(E[ptr[0]:ptr[1]], I[ptr[0]:ptr[1]], fs=fs, window='hann', nperseg=None, noverlap=None, nfft=nfft)
    f, Cny2h2p[jj, :] = csd(N[ptr[0]:ptr[1]], I[ptr[0]:ptr[1]], fs=fs, window='hann', nperseg=None, noverlap=None, nfft=nfft)
    f, Cee2h2p[jj, :] = csd(E[ptr[0]:ptr[1]], E[ptr[0]:ptr[1]], fs=fs, window='hann', nperseg=None, noverlap=None, nfft=nfft)
    f, Cnn2h2p[jj, :] = csd(N[ptr[0]:ptr[1]], N[ptr[0]:ptr[1]], fs=fs, window='hann', nperseg=None, noverlap=None, nfft=nfft)
    f, Cne2h2p[jj, :] = csd(N[ptr[0]:ptr[1]], E[ptr[0]:ptr[1]], fs=fs, window='hann', nperseg=None, noverlap=None, nfft=nfft)
    f, Cyy2h2p[jj, :] = csd(I[ptr[0]:ptr[1]], I[ptr[0]:ptr[1]], fs=fs, window='hann', nperseg=None, noverlap=None, nfft=nfft)

    # Angle loop
    countr = 0

    # Loop over all azimuths
    


# Load synthetic data
from umbra.IO import read_files as rf
import umbra.misc.sorting as ss
import umbra.plotting.stream_plots as sp
from obspy.geodetics.base import gps2dist_azimuth

# SPECFEM
# dir = 'TOPO83'
# srcdir = '/Users/jwbishop/Documents/SPECFEM_Runs/'+dir
# srcdir1 = srcdir+'/mseed'
# files = rf.get_files(srcdir1, '.mseed')

# HAR JP 3496531.68687 655051.39768 0.0 2890.0
#latorUTM:       3494942
#longorUTM:      657993
# dy = 3496531.68687 - 3494942
# dx = 655051.39768 - 657993
# np.arctan2(dy, dx)*180/np.pi

# st = rf.make_stream(srcdir1, ['300_'+dir+'.mseed'])
# st = read('/Users/jwbishop/Documents/SPECFEM_RUNS/TOPO60/HAR_TOPO60.mseed')
# st.resample(20)
# st.filter('bandpass', freqmin=FMIN, freqmax=FMAX, corners=2, zerophase=True)
# st.taper(max_percentage=0.01)
# st.sort(['channel'], reverse=False)

# Remove particle velocity traces
for jj in range(0, 3):
    st.remove(st[1])

st.sort(['channel'], reverse=True)

# Rename X & Y to E & N
for tr in st:
    if tr.stats.channel[-1] == 'X':
        tr.stats.channel = 'FXE'
    elif tr.stats.channel[-1] == 'Y':
        tr.stats.channel = 'FXN'

st.plot()

# %% Grab and filter waveforms
st = gather_waveforms(SOURCE, NETWORK, STATION, LOCATION, CHANNEL, START, END,
                      remove_response=True)
stf = st.copy()
stf.filter('bandpass', freqmin=FMIN, freqmax=FMAX, corners=2, zerophase=True)
stf.taper(max_percentage=0.1)
stf.rotate('NE->RT', back_azimuth=133)
stf.normalize()
stf.plot()

winlen = WINLEN
winover = WINOVER
fmin = FMIN
fmax = FMAX

# Put in Z, N, E order
# stf.sort(['channel'], reverse=False)
# stf.remove(stf[3])

# tr = stf[1]
# tr2 = stf[2]
# st = stf.copy()
# st.remove(tr)
# st.remove(tr2)

# Pull processing parameters from the stream file.
tvec = st[0].times('matplotlib')
nchans = len(st)
npts = st[0].stats.npts
fs = st[0].stats.sampling_rate

# check that all traces have the same length
if len(set([len(tr) for tr in st])) != 1:
    raise ValueError('Traces in stream must have same length!')

# Convert window length to samples
winlensamp = int(winlen*fs)
sampinc = int((1-winover)*winlensamp)
its = np.arange(0, npts, sampinc)
# We need all the data windows to be the same length
# The last data block is discarded.
nits = len(its) - int(np.ceil(winlensamp/sampinc))

# Pre-allocate data arrays.
t = np.full(nits, np.nan)
baz = np.full((nits,), np.nan)

# Loop through the time series and rotate the components
az_test = np.linspace(0, 360, 720)
drot = 0.5
counter = 0

for jj in range(nits):
    # Get time from middle of window, except for the end.
    ptr = int(its[jj]), int(its[jj] + winlensamp)
    try:
        t[jj] = tvec[ptr[0]+int(winlensamp/2)]
    except:
        t[jj] = np.nanmax(t, axis=0)

    test_baz = np.fill(len(az_test), np.nan)

    for kk in range(0, len(az_test)):
        st.rotate(drot)
        # Store data traces in an array for processing.
        data = np.empty((npts, nchans))
        for ii in range(0, nchans):
            data[:, ii] = st[ii].data

        # Calculate the cross-spectral matrix S
        S, freq = pa.calc_csm(data[ptr[0]:ptr[1], :], fs, fmin, fmax)

        # Calculate the coherence and phase spectrum
        gtest, _ = pa.calc_gp(S, freq)
        test_baz[kk] = np.mean(gtest)

    # Print progress
    baz[jj] = np.argmin(test_baz)*drot
    counter += 1
    print('{:.1f}%'.format((counter / nits) * 100), end='\r')




# Loop through the time series and rotate the components
baz_test = np.linspace(0, 360, 721)
counter = 0


jj = 0
for jj in range(0, nits):
    # Get time from middle of window, except for the end.
    ptr = int(its[jj]), int(its[jj] + winlensamp)
    try:
        t[jj] = tvec[ptr[0]+int(winlensamp/2)]
    except:
        t[jj] = np.nanmax(t, axis=0)

    gmed = np.full(len(baz_test), np.nan)
    for kk in range(0, len(baz_test)):
        st_rot = st.copy()
        st_rot.rotate(method='NE->RT', back_azimuth=baz_test[kk])
        st_rot.remove(st_rot[1])
        st_rot.remove(st_rot[2])
        st_rot.sort(['channel'], reverse=False)
        # st_rot[1].plot()

        # Store data traces in an array for processing.
        data = np.empty((npts, 2))
        for ii in range(0, 2):
            data[:, ii] = st_rot[ii].data

        # Calculate the cross-spectral matrix S
        S, freq = pa.calc_csm(data[ptr[0]:ptr[1], :], fs, fmin, fmax)

        # Calculate the coherence and phase spectrum
        gtest, p = pa.calc_gp(S, freq)
        gmed[kk] = np.mean(gtest)

    # Print progress
    idx = np.argmin(gmed)
    baz[jj] = baz_test[idx]
    counter += 1
    print('{:.1f}%'.format((counter / nits) * 100), end='\r')

print('\nDone\n')

st_rot

baz

fig, axs = plt.subplots(2, 1, sharex='col')
fig.set_size_inches(10, 9)
axs[0].plot(tvec, st[0].data)
axs[1].scatter(t, baz)
axs[1].axhline(y=90)
# axs[1].axhline(y=0+180)
axs[1].xaxis_date()
axs[1].set_ylabel('Baz [degree]')
axs[1].set_ylim(0, 360)
axs[1].tick_params(axis='x', labelbottom='on')
axs[1].fmt_xdata = dates.DateFormatter('%HH:%MM')
axs[1].xaxis.set_major_formatter(dates.DateFormatter("%d-%H:%M"))
axs[1].set_xlabel('UTC Time')



# stf = read('/Users/jwbishop/Documents/SPECFEM_RUNS/TOPO107/HAR_TOPO107.mseed')
# stf.filter('bandpass', freqmin=FMIN, freqmax=FMAX, corners=2, zerophase=True)
# stf.taper(max_percentage=0.01)

# Put in Z, N, E order
stf.sort(['channel'], reverse=True)
stf
tr = stf[1]
tr2 = stf[2]
tr3 = stf[3]
tr3.plot()
plt.plot(stf[3].data)
z = stf[0].data[50:92]
x = stf[2].data[50:92]
y = stf[2].data
t = np.array(np.arange(0, len(z)))
plt.scatter(z, x, c=t)
plt.colorbar()


N = len(z)
X = x
Y = z
dt = 0.02
T = N
w = 2*np.pi/T
arr1 = np.vstack((X, Y))
arr2 = np.concatenate((np.reshape(np.cos(w*t), (len(z), 1)), np.reshape(np.sin(w*t), (len(z), 1))), axis=1)
M = (2/N)*arr1@arr2
M
q = -M[0, 1]/M[1, 1]
s = 1 + q**2
x1 = np.sqrt(2*s**2 + 2*s*q)
x2 = np.sqrt(2*s**2 - 2*s*q)
print(x1)
print(x2)

c = 340/np.cos(40/180*np.pi)
B = c/x2
print(B)

# Formulate and solve the least squares problem
# ||Ax - b ||^2
# https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
A = np.vstack([x**2, x*z, z**2, x, z]).T
bb = np.ones_like(x)
x = np.linalg.lstsq(A, bb, rcond=None)[0].squeeze()
A = x[0]
B = x[1]
C = x[2]
D = x[3]
E = x[4]
F = 0
M = np.array([[F, D/2, E/2], [D/2, A, B/2], [E/2, B/2 C]])
M0 = np.array([[A, B/2], [B/2, C]])
ang = np.arctan2(B, (A-C))/2 * 180/np.pi

c = 340
b = np.linspace(100, c, 1000)
phi = np.arctan2((2 - (c/b)**2), (2 * np.sqrt((c/b)**2 - 1))) * 180/np.pi
plt.plot(b, phi)
plt.axhline(y=ang)

# Convert from velocity to displacement
stf.integrate()

tr = stf[1]
tr2 = stf[2]
tr3 = stf[3]
tr4 = stf[4]
tr5 = stf[5]
stf.remove(tr)
stf.remove(tr2)
stf.remove(tr3)
stf.remove(tr4)
stf.remove(tr5)

# stf[0].resample(20)
# stf[1].resample(20)
# stf[2].resample(20)

stf.sort(['channel'])
stf.plot()

# Rotate?
# Make this a new function
# cmtlat = 52.0497
# cmtlon = -176.0788
# latlist = [tr.stats.latitude for tr in stf]
# lonlist = [tr.stats.longitude for tr in stf]
# stlat = np.mean(latlist)
# stlon = np.mean(lonlist)
# dist, az, baz = obs.gps2dist_azimuth(cmtlat, cmtlon, stlat, stlon)
# for jj in range(0, 180):
#     stf.rotate(method='NE->RT', back_azimuth=jj)
#     stf.plot()

stf.plot()
z = stf.traces[0].data[15:50]
x = stf.traces[1].data[15:50]
t = np.array(np.arange(0, len(z)))
plt.plot(x, z)
plt.colorbar()

plt.scatter(x, z, c=t)
plt.colorbar()


# Convert data to RTZ or just specify components
# Requires data be in ZNE order.

# Time Frequency polarization analysis
t, freq, rect, plan, ellip, incid, az = polartf(stf, WINLEN, WINOVER, FMIN, FMAX)

# Coherence Analysis
t, freq, gamma2, phase2 = pcohere(stf, WINLEN, WINOVER, FMIN, FMAX)



# Plotting
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np

from copy import deepcopy
from collections import Counter

# Specify the colormap.
cm = cc.cm.rainbow
# Colorbar/y-axis limits for MdCCM.
cax = (0.2, 1)
# Specify the time vector for plotting the trace.
tvec = stf[0].times('matplotlib')

# Start Plotting.
# Initiate and plot the trace.
fig, axs = plt.subplots(3, 1, sharex='col')
fig.set_size_inches(10, 9)
axs[0].plot(tvec, stf[1].data*1e6, 'k')
axs[0].set_xlim(t[0], t[-1])
axs[0].axis('tight')
axs[0].set_ylabel('Displacement \n [micro-m]')

# Plot the Incidence Angle
sc = axs[1].pcolormesh(t, freq, gamma2, cmap=cm, vmin=0, vmax=1)
axs[1].axis('tight')
axs[1].set_xlim(t[0], t[-1])
axs[1].set_ylim(freq[0], freq[-1])
# sc.set_clim((0.1, 180))
p1 = axs[1].get_position()
cbaxes1 = fig.add_axes([0.92, p1.y0, 0.02, p1.height])
hc1 = plt.colorbar(sc, orientation="vertical",
                   cax=cbaxes1, ax=axs[1])
# hc1.set_label('Incidence (deg)')
hc1.set_label('Gamma^2')
axs[1].set_ylabel('Frequency [Hz]', fontsize=12)

# Plot the Azimuth
sc = axs[2].pcolormesh(t, freq, phase2, cmap=cm)
axs[2].axis('tight')
axs[2].set_xlim(t[0], t[-1])
axs[2].set_ylim(freq[0], freq[-1])
# sc.set_clim((0.01, 100))
p2 = axs[2].get_position()
cbaxes2 = fig.add_axes([0.92, p2.y0, 0.02, p2.height])
hc2 = plt.colorbar(sc, orientation="vertical",
                   cax=cbaxes2, ax=axs[2])
# hc2.set_label('Azimuth (deg)')
hc2.set_label('Phase (deg)')
axs[2].set_ylabel('Frequency [Hz]', fontsize=12)

axs[2].xaxis_date()
axs[2].tick_params(axis='x', labelbottom='on')
axs[2].fmt_xdata = dates.DateFormatter('%HH:%MM')
axs[2].xaxis.set_major_formatter(dates.DateFormatter("%d-%H:%M"))
axs[2].set_xlabel('UTC Time')


# Plot the Rectilinearity
sc = axs[3].pcolormesh(t, freq, rect**2, cmap=cm)
axs[3].axis('tight')
axs[3].set_xlim(t[0], t[-1])
axs[3].set_ylim(freq[0], freq[-1])
sc.set_clim((0.01, 1.0))
p3 = axs[3].get_position()
cbaxes3 = fig.add_axes([0.92, p3.y0, 0.02, p3.height])
hc3 = plt.colorbar(sc, orientation="vertical",
                   cax=cbaxes3, ax=axs[3])
hc3.set_label('Rectilinearity^2')
axs[3].set_ylabel('Frequency [Hz]', fontsize=12)

# Plot the Ellipticity
sc = axs[4].pcolormesh(t, freq, ellip, cmap=cm)
axs[4].axis('tight')
axs[4].set_xlim(t[0], t[-1])
axs[4].set_ylim(freq[0], freq[-1])
sc.set_clim((0.01, 1.0))
p4 = axs[4].get_position()
cbaxes4 = fig.add_axes([0.92, p4.y0, 0.02, p4.height])
hc4 = plt.colorbar(sc, orientation="vertical",
                   cax=cbaxes4, ax=axs[4])
hc4.set_label('Ellipticity')
axs[4].set_ylabel('Frequency [Hz]', fontsize=12)

# Plot the Planarity
# sc = axs[5].pcolormesh(t, freq, plan, cmap=cm)
# axs[5].axis('tight')
# axs[5].set_xlim(t[0], t[-1])
# axs[5].set_ylim(freq[0], freq[-1])
# sc.set_clim((0.01, 1.0))
# p5 = axs[5].get_position()
# cbaxes5 = fig.add_axes([0.92, p5.y0, 0.02, p5.height])
# hc5 = plt.colorbar(sc, orientation="vertical",
#                    cax=cbaxes5, ax=axs[5])
# hc5.set_label('Planarity')
# axs[5].set_ylabel('Frequency [Hz]', fontsize=12)

axs[4].xaxis_date()
axs[4].tick_params(axis='x', labelbottom='on')
axs[4].fmt_xdata = dates.DateFormatter('%HH:%MM')
axs[4].xaxis.set_major_formatter(dates.DateFormatter("%d-%H:%M"))
axs[4].set_xlabel('UTC Time')
# fig.savefig('GS_time_freq_polarization.png', bbox_inches='tight')

import os as os
os.getcwd()
