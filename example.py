# import colorcet as cc
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
from obspy.core import UTCDateTime
import obspy.geodetics.base as obs
from obspy import read

from scipy.signal import csd
from scipy.fft import rfftfreq

from waveform_collection import gather_waveforms
# from tcm.algorithms import tcm

import os as os
os.chdir('/Users/jwbishop/Desktop/Matt_TCM_Code')

# Read example data
st = read('TCM_Example_GSMY.mseed')
tvec = st[0].times('matplotlib')


num_chans = len(st)
num_pts = st[0].stats.npts

data = np.zeros((num_pts, num_chans))
for ii, tr in enumerate(st):
    data[:, ii] = tr.data
sampling_rate = st[0].stats.sampling_rate

# Filter range [Hz]
freq_min = 10.0
freq_max = 20.0

# Window length [sec]
window_length = 4
# Window size [samples]
WIND = window_length * sampling_rate
# Amount of window overlap
WINOVER = 0.10
WINDOW = WIND/2

nfft = np.power(2, int(np.ceil(np.log2(WIND))))
fss = sampling_rate

# Data Mapping
Infra = data[:, 0]
E = data[:, 3]
N = data[:, 2]
Z = data[:, 1]

# Matt's `wind`
winlensamp = int(WIND)
# 90% overlap
winover = 0.90
window = winlensamp/2
sampinc = int((1 - winover) * winlensamp) + 1
npts = len(Infra) - winlensamp
its = np.arange(0, npts, sampinc)
nits = len(its)

freq = rfftfreq(nfft, 1/sampling_rate)
# freq = freq[np.where((freq >= FMIN) & (freq <= FMAX))]

Cxy2_zi = np.empty((nits, len(freq)), dtype=np.complex128)
Cxy2_ii = np.empty((nits, len(freq)), dtype=np.complex128)
Cxy2_zz = np.empty((nits, len(freq)), dtype=np.complex128)

# Compute the vertical coherence
# For every time window, calculate the cross power
# spectral density at every frequency
t = np.full(nits, np.nan)
for jj in range(0, nits):
    # Get time from middle of window, except for the end.
    ptr = [int(its[jj]), int(its[jj] + winlensamp)]
    try:
        t[jj] = tvec[ptr[0]+int(winlensamp/2)]
    except:
        t[jj] = np.nanmax(t, axis=0)

    _, Cxy2_zi[jj, :] = csd(
        Z[ptr[0]:ptr[1]], Infra[ptr[0]:ptr[1]],
        fs=sampling_rate, window='hann',
        nperseg=nfft, noverlap=None, nfft=nfft)
    _, Cxy2_ii[jj, :] = csd(
        Infra[ptr[0]:ptr[1]], Infra[ptr[0]:ptr[1]], fs=sampling_rate, window='hann',
        nperseg=nfft, noverlap=None, nfft=nfft)
    _, Cxy2_zz[jj, :] = csd(
        Z[ptr[0]:ptr[1]], Z[ptr[0]:ptr[1]],
        fs=sampling_rate, window='hann',
        nperseg=nfft, noverlap=None, nfft=nfft)

# Calculate the normalized coherence between the vertical
# seismic channel and the infrasound channel
Cxy2 = np.real(np.multiply(Cxy2_zi, np.conjugate(Cxy2_zi)) / np.multiply(Cxy2_ii, Cxy2_zz))
# Calculate the gain between the cross spectrum and the infrasound
Cxy2_g = np.abs(Cxy2_zi) /np.real(Cxy2_ii)

# Time Vector
tvecC = its/sampling_rate

# Add half the window length [sec]
tvecC_z = tvecC + window_length/2

# Azimuths to scan over [degreem]
azvect = np.array(np.arange(-179, 181, 1))

# Get the closest frequency points to the preferred ones
fmin_ind = np.argmin(np.abs(freq_min - freq))
fmax_ind = np.argmin(np.abs(freq_max - freq))


Cey2h2p = np.empty((nits, len(freq)), dtype=np.complex128)
Cny2h2p = np.empty((nits, len(freq)), dtype=np.complex128)
Cee2h2p = np.empty((nits, len(freq)), dtype=np.complex128)
Cnn2h2p = np.empty((nits, len(freq)), dtype=np.complex128)
Cne2h2p = np.empty((nits, len(freq)), dtype=np.complex128)
Cyy2h2p = np.empty((nits, len(freq)), dtype=np.complex128)
Cxy2h2 = np.empty((nits, len(freq)), dtype=np.complex128)
s2mw = np.empty((nits, len(azvect)), dtype=np.complex128)
s22mw = np.empty((nits, len(azvect)), dtype=np.complex128)
# Loop over windows
# Calculate the cross spectrum at every frequency

for jj in range(0, nits):
    ptr = [int(its[jj]), int(its[jj] + winlensamp)]
    # We only need 7 cross spectra for the angle sum
    _, Cey2h2p[jj, :] = csd(
        E[ptr[0]:ptr[1]], Infra[ptr[0]:ptr[1]],
        fs=sampling_rate, window='hann',
        nperseg=nfft, noverlap=None, nfft=nfft)
    _, Cny2h2p[jj, :] = csd(
        N[ptr[0]:ptr[1]], Infra[ptr[0]:ptr[1]],
        fs=sampling_rate, window='hann',
        nperseg=nfft, noverlap=None, nfft=nfft)
    _, Cee2h2p[jj, :] = csd(
        E[ptr[0]:ptr[1]], E[ptr[0]:ptr[1]],
        fs=sampling_rate, window='hann',
        nperseg=nfft, noverlap=None, nfft=nfft)
    _, Cnn2h2p[jj, :] = csd(
        N[ptr[0]:ptr[1]], N[ptr[0]:ptr[1]],
        fs=sampling_rate, window='hann',
        nperseg=nfft, noverlap=None, nfft=nfft)
    _, Cne2h2p[jj, :] = csd(
        N[ptr[0]:ptr[1]], E[ptr[0]:ptr[1]],
        fs=sampling_rate, window='hann',
        nperseg=nfft, noverlap=None, nfft=nfft)
    _, Cyy2h2p[jj, :] = csd(
        Infra[ptr[0]:ptr[1]], Infra[ptr[0]:ptr[1]],
        fs=sampling_rate, window='hann',
        nperseg=nfft, noverlap=None, nfft=nfft)

    # Angle loop
    countr = -1
    # Loop over all azimuths
    for azs in azvect:
        countr += 1
        # Calculate transverse coherence for the trial azimuth
        Cxy2h2[jj, :] = (np.abs(Cny2h2p[jj, :] * np.sin(azs * np.pi/180) - Cey2h2p[jj, :] * np.cos(azs*np.pi/180))**2) / (np.abs(Cnn2h2p[jj, :] * np.sin(azs * np.pi/180)**2 - 2 * np.real(Cne2h2p[jj, :]) * np.sin(azs * np.pi/180) * np.cos(azs * np.pi/180) + Cee2h2p[jj, :] * np.cos(azs * np.pi/180) * np.cos(azs * np.pi/180)) * np.abs(Cyy2h2p[jj, :]))

    # Weighting potential transverse coherence based on vertical coherence
    s2mw[jj, countr] = np.sum(
        Cxy2h2[jj, fmin_ind:fmax_ind] * Cxy2[jj, fmin_ind:fmax_ind])
    # Sum of vertical coherence for denominator of weighted sum
    s22mw[jj, countr] = np.sum(Cxy2[jj, fmin_ind:fmax_ind])


# Save the extent of the frequency band of interest in indicies
bbf1 = fmin_ind
bbf2 = fmax_ind

# Form the weighted sum that weights transverse coherence by vertical coherence
dum = s2mw/s22mw

# Apply some smoothing if desired
# Number of samples in coherogram to smooth
nsmth = 4
bbv = np.full(nits, np.nan)
bbv2 = np.full(nits, np.nan)
aa2 = np.full(nits, np.nan)
bb2 = np.full(nits, np.nan)
for jj in range(0, len(dum) - nsmth):
    # Here are the 2 possible back-azimuths
    idx = np.argsort(np.sum(dum[:, jj:(jj + nsmth)], 1))
    bbv[jj] = idx[0]
    bbv2[jj] = idx[1]
    # Info on the amount of coherence
    aa2[jj] = np.max(np.mean(dum[:, jj:(jj + nsmth)], 1))
    bb2[jj] = np.argmax(np.mean(dum[:, jj:(jj + nsmth)], 1))

# Resolve the 180 degree ambiguity


#######################
# Plotting
#######################
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 1, sharex='col')
axs[0].plot(tvec, Infra, c='k')
axs[1].pcolormesh(t, freq, Cxy2.T)
axs[2].scatter(t, aa2)
axs[2].xaxis_date()
axs[2].tick_params(axis='x', labelbottom='on')
axs[2].fmt_xdata = dates.DateFormatter('%HH:%MM')
axs[2].xaxis.set_major_formatter(dates.DateFormatter("%d-%H:%M"))
axs[2].set_xlabel('UTC Time')
# fig.savefig('TCM_Example.png', bbox_inches='tight', dpi=300, facecolor="w")