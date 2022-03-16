import colorcet as cc
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
# from obspy.core import UTCDateTime
# import obspy.geodetics.base as obs
from obspy import read

from scipy.signal import csd, windows
from scipy.fft import rfftfreq

import os as os
# os.chdir('/Users/jwbishop/Desktop/Matt_TCM_Code')
os.chdir('/Users/jwbishop/Documents/Github/tcm')

import tcm_classes

# from waveform_collection import gather_waveforms
# from tcm.algorithms import tcm

# Read example data
st = read('/Users/jwbishop/Desktop/Matt_TCM_Code/TCM_Example_GSMY.mseed')

# Filter range [Hz]
freq_min = 10.0
freq_max = 20.0

# Window length [sec]
window_length = 4.0

# Window Overlap [0.0, 1.0)
window_overlap = 0.90

# Azimuths to scan over [degrees]
az_min = -179
az_max = 180
az_delta = 1

# Use retrograde motion to determine back-azimuth angle
assume_retrograde = True

################
# End User Input
################

# Create object to hold data and pre-process
data = tcm_classes.DataBin(freq_min, freq_max, window_length, window_overlap, az_min, az_max, az_delta, assume_retrograde)
data.build_data_arrays(st)

# Create cross-spectral matrix object
CSM = tcm_classes.Spectral(data)
# Calculate the vertical coherence
CSM.calculate_vertical_Cxy2(data)
# Calculate the transverse coherence over all trial azimuths
CSM.calculate_tcm_over_azimuths(data)
# Find the coherence minima and apply the retrograde assumption if applicable
az = CSM.find_minimum_tc(data)
print(az)

#######################
# Plotting
#######################
fig, axs = plt.subplots(3, 1, sharex='col')
# Trace
axs[0].plot(data.tvec, data.Z, c='k')
axs[0].set_ylabel('Displacement \n [m]')
# Gamma^2
sc0 = axs[1].pcolormesh(CSM.t, CSM.freq_vector, CSM.Cxy2, cmap=cc.cm.rainbow)
axs[1].axis('tight')
axs[1].set_xlim(CSM.t[0], CSM.t[-1])
axs[1].set_ylim(CSM.freq_vector[0], CSM.freq_vector[-1])
axs[1].set_ylabel('Frequency \n [Hz]', fontsize=12)
p1 = axs[1].get_position()
cbaxes1 = fig.add_axes([0.92, p1.y0, 0.02, p1.height])
hc1 = plt.colorbar(sc0, orientation="vertical",
                   cax=cbaxes1, ax=axs[1])
hc1.set_label('Max Weighted \n Coherence')
sc0.set_clim(0.0, 1.0)


# Back-azimuth Estimate
sc1 = axs[2].scatter(CSM.t[CSM.smvc], az - 181, c=CSM.aa2, cmap=cc.cm.rainbow)
# axs[2].set_ylim(-180, 180)
axs[2].axhline(-52)
axs[2].set_ylim(-180, 180)
axs[2].set_ylabel('Back-Azimuth \n [Deg.]', fontsize=12)
p1 = axs[2].get_position()
cbaxes1 = fig.add_axes([0.92, p1.y0, 0.02, p1.height])
hc1 = plt.colorbar(sc1, orientation="vertical",
                   cax=cbaxes1, ax=axs[2])
hc1.set_label('Max Weighted \n Coherence')
sc1.set_clim(0.0, 1.0)

axs[2].xaxis_date()
axs[2].tick_params(axis='x', labelbottom='on')
axs[2].fmt_xdata = dates.DateFormatter('%HH:%MM')
axs[2].xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
axs[2].set_xlabel('UTC Time')
# fig.savefig('Python_TCM_Example.png', bbox_inches='tight', dpi=300, facecolor="w") # noqa



# Data items
# tvec = st[0].times('matplotlib')
# num_chans = len(st)
# num_pts = st[0].stats.npts
# data = np.zeros((num_pts, num_chans))
# for ii, tr in enumerate(st):
#     data[:, ii] = tr.data
# sampling_rate = st[0].stats.sampling_rate

# Data Mapping
# Infra = data[:, 0]
# Z = data[:, 1]
# N = data[:, 2]
# E = data[:, 3]


# Window size [samples]
# winlensamp = int(data.window_length * data.sampling_rate)

# Amount of window overlap - 90%
# winover = 0.90
# sub_window = winlensamp/2
# sampinc = int((1 - winover) * winlensamp) + 1
# npts = len(data.Infra) - winlensamp
# its = np.arange(0, npts, sampinc)
# nits = len(its)

# Spectral Object
# freq_vector= rfftfreq(nfft, 1/data.sampling_rate)

# Cxy2_zi = np.empty((len(freq), nits), dtype=np.complex128)
# Cxy2_ii = np.full((len(freq), nits), np.nan)
# Cxy2_zz = np.full((len(freq), nits), np.nan)
# Cxy2 = np.full((len(freq), nits), np.nan)
# Cxy2_g = np.full((len(freq), nits), np.nan)

# Compute the vertical coherence - method function
# For every time window, calculate the cross power
# spectral density at every frequency
# t = np.full(nits, np.nan)
# # win = windows.hamming(int(window), sym=False)
# for jj in range(0, nits):
#     # Get time from middle of window, except for the end.
#     ptr = [int(its[jj]), int(its[jj] + winlensamp)]
#     try:
#         t[jj] = tvec[ptr[0]+int(winlensamp/2)]
#     except:
#         t[jj] = np.nanmax(t, axis=0)
#
#     _, Cxy2_zi[:, jj] = csd(
#         data.Z[ptr[0]:ptr[1]], data.Infra[ptr[0]:ptr[1]],
#         fs=data.sampling_rate, window=win,
#         nperseg=sub_window, noverlap=50, nfft=nfft)
#     _, Cxy2_ii[:, jj] = np.real(csd(
#         data.Infra[ptr[0]:ptr[1]], data.Infra[ptr[0]:ptr[1]],
#         fs=data.sampling_rate, window=win,
#         nperseg=sub_window, noverlap=50, nfft=nfft))
#     _, Cxy2_zz[:, jj] = np.real(csd(
#         data.Z[ptr[0]:ptr[1]], data.Z[ptr[0]:ptr[1]],
#         fs=data.sampling_rate, window=win,
#         nperseg=sub_window, noverlap=50, nfft=nfft))
#
#     # Calculate the normalized coherence between the vertical
#     # seismic channel and the infrasound channel
#     Cxy2[:, jj] = np.real(np.multiply(Cxy2_zi[:, jj], np.conjugate(Cxy2_zi[:, jj]))) / np.multiply(Cxy2_ii[:, jj], Cxy2_zz[:, jj]) # noqa
#     # Calculate the gain between the cross spectrum and the infrasound
#     # Cxy2_g[:, jj] = np.abs(Cxy2_zi[:, jj]) / np.real(Cxy2_ii[:, jj])

# Time Vector
# tvecC = its/sampling_rate

# Add half the window length [sec]
# tvecC_z = tvecC + window_length/2

# # Get the closest frequency points to the preferred ones
# fmin_ind = np.argmin(np.abs(freq_min - CSM.freq_vector))
# fmax_ind = np.argmin(np.abs(freq_max - CSM.freq_vector))

# # Create azimuth vector [degrees]
# azvect = np.array(np.arange(az_min, az_max + az_delta, az_delta))

# Cey2h2p = np.empty((len(CSM.freq_vector), data.nits), dtype=np.complex128)
# Cny2h2p = np.empty((len(CSM.freq_vector), data.nits), dtype=np.complex128)
# Cee2h2p = np.empty((len(CSM.freq_vector), data.nits), dtype=np.complex128)
# Cnn2h2p = np.empty((len(CSM.freq_vector), data.nits), dtype=np.complex128)
# Cne2h2p = np.empty((len(CSM.freq_vector), data.nits), dtype=np.complex128)
# Cyy2h2p = np.empty((len(CSM.freq_vector), data.nits), dtype=np.complex128)
# Cxy2h2 = np.empty((len(CSM.freq_vector), data.nits))
# s2mw = np.empty((len(azvect), data.nits))
# s22mw = np.empty((len(azvect), data.nits))

# # Loop over windows
# # Calculate the cross spectrum at every frequency
# for jj in range(0, data.nits):
#     ptr = [int(its[jj]), int(its[jj] + winlensamp)]
#     # We only need 7 cross spectra for the angle sum
#     _, Cey2h2p[:, jj] = csd(
#         data.E[ptr[0]:ptr[1]], data.Infra[ptr[0]:ptr[1]],
#         fs=data.sampling_rate, window=window,
#         nperseg=sub_window, noverlap=50, nfft=nfft)
#     _, Cny2h2p[:, jj] = csd(
#         data.N[ptr[0]:ptr[1]], data.Infra[ptr[0]:ptr[1]],
#         fs=data.sampling_rate, window=window,
#         nperseg=sub_window, noverlap=50, nfft=nfft)
#     _, Cee2h2p[:, jj] = csd(
#         data.E[ptr[0]:ptr[1]], data.E[ptr[0]:ptr[1]],
#         fs=data.sampling_rate, window=window,
#         nperseg=sub_window, noverlap=50, nfft=nfft)
#     _, Cnn2h2p[:, jj] = csd(
#         data.N[ptr[0]:ptr[1]], data.N[ptr[0]:ptr[1]],
#         fs=data.sampling_rate, window=window,
#         nperseg=sub_window, noverlap=50, nfft=nfft)
#     _, Cne2h2p[:, jj] = csd(
#         data.N[ptr[0]:ptr[1]], data.E[ptr[0]:ptr[1]],
#         fs=data.sampling_rate, window=window,
#         nperseg=sub_window, noverlap=50, nfft=nfft)
#     _, Cyy2h2p[:, jj] = csd(
#         data.Infra[ptr[0]:ptr[1]], data.Infra[ptr[0]:ptr[1]],
#         fs=data.sampling_rate, window=window,
#         nperseg=sub_window, noverlap=50, nfft=nfft)

#     # Loop over all azimuths
#     for kk in range(0, len(azvect)):
#         # Calculate transverse coherence for the trial azimuth
#         Cxy2h2[:, jj] = (np.abs(Cny2h2p[:, jj] * np.sin(azvect[kk] * np.pi/180) - Cey2h2p[:, jj] * np.cos(azvect[kk] * np.pi/180))**2) / (np.abs(Cnn2h2p[:, jj] * np.sin(azvect[kk] * np.pi/180)**2 - 2 * np.real(Cne2h2p[:, jj]) * np.sin(azvect[kk] * np.pi/180) * np.cos(azvect[kk] * np.pi/180) + Cee2h2p[:, jj] * np.cos(azvect[kk] * np.pi/180)**2) * np.abs(Cyy2h2p[:, jj])) # noqa

#         # Weighting potential transverse coherence based on vertical coherence
#         s2mw[kk, jj] = np.sum(
#             Cxy2h2[fmin_ind:fmax_ind, jj] * Cxy2[fmin_ind:fmax_ind, jj])
#         # Sum of vertical coherence for denominator of weighted sum
#         s22mw[kk, jj] = np.sum(Cxy2[fmin_ind:fmax_ind, jj])

# # Save the extent of the frequency band of interest in indicies
# bbf1 = fmin_ind
# bbf2 = fmax_ind

# Form the weighted sum that weights transverse coherence by vertical coherence
# dum = s2mw/s22mw

"""
fig, axs = plt.subplots(1, 1, sharex='col')
sc0 = axs.pcolormesh(t, azvect, dum, cmap=cc.cm.rainbow)
axs.axis('tight')
axs.set_xlim(t[0], t[-1])
axs.set_ylim(azvect[0], azvect[-1])
axs.set_ylabel('Frequency \n [Hz]', fontsize=12)
p1 = axs.get_position()
cbaxes1 = fig.add_axes([0.92, p1.y0, 0.02, p1.height])
hc1 = plt.colorbar(sc0, orientation="vertical",
                   cax=cbaxes1, ax=axs)
hc1.set_label('Val.')
# sc0.set_clim(0.0, 1.0)
axs.xaxis_date()
axs.tick_params(axis='x', labelbottom='on')
axs.fmt_xdata = dates.DateFormatter('%HH:%MM')
axs.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
axs.set_xlabel('UTC Time')
"""
#
# # Apply some smoothing if desired
# # Number of samples in coherogram to smooth
# nsmth = 4
# bbv = np.full(data.nits - nsmth, 0, dtype='int')
# bbv2 = np.full(data.nits - nsmth, 0, dtype='int')
# aa2 = np.full(data.nits - nsmth, np.nan)
# bb2 = np.full(data.nits - nsmth, 0, dtype='int')
#
# # Here are the 2 possible back-azimuths
# for jj in range(0, data.nits - nsmth):
#     idx = np.argsort(np.sum(dum[:, jj:(jj + nsmth + 1)], 1))
#     bbv[jj] = idx[0]
#     bbv2[jj] = idx[1]
#     # Info on the amount of coherence
#     aa2[jj] = np.max(np.mean(dum[:, jj:(jj + nsmth + 1)], 1))
#     bb2[jj] = np.argmax(np.mean(dum[:, jj:(jj + nsmth + 1)], 1))
#
#
# # Resolve the 180 degree ambiguity
# # Make this a flag that defaults to `True`
# Cxy2rz = np.empty((len(CSM.freq_vector), data.nits), dtype=np.complex128)
# Cxy2rz2 = np.empty((len(CSM.freq_vector), data.nits), dtype=np.complex128)
# Cxy2rza = np.empty((len(CSM.freq_vector), data.nits), dtype=np.complex128)
# Cxy2rza2 = np.empty((len(CSM.freq_vector), data.nits), dtype=np.complex128)
# # for jj in range(((nsmth/2) + 1), (nits - (nsmth/2))):
# for jj in range(0, data.nits - nsmth):
#     ptr = [int(its[jj]), int(its[jj] + winlensamp)]
#     _, Cxy2rz[:, jj] = csd(
#         data.Z[ptr[0]:ptr[1]], data.N[ptr[0]:ptr[1]] * np.cos(
#             azvect[bbv[jj]] * np.pi/180) + data.E[ptr[0]:ptr[1]] * np.sin(
#                 azvect[bbv[jj]] * np.pi/180),
#         fs=data.sampling_rate, window=window,
#         nperseg=sub_window, noverlap=50, nfft=nfft) # noqa
#     _, Cxy2rz2[:, jj] = csd(
#         data.Z[ptr[0]:ptr[1]], data.N[ptr[0]:ptr[1]] * np.cos(
#             azvect[bbv2[jj]] * np.pi/180) + data.E[ptr[0]:ptr[1]] * np.sin(
#                 azvect[bbv2[jj]] * np.pi/180),
#         fs=data.sampling_rate, window=window,
#         nperseg=sub_window, noverlap=50, nfft=nfft) # noqa
#     Cxy2rza[:, jj] = np.angle(Cxy2rz[:, jj])
#     Cxy2rza2[:, jj] = np.angle(Cxy2rz2[:, jj])
# # The time vector for the case of nonzero smoothing
# smvc = np.arange(((nsmth/2) + 1), (data.nits - (nsmth/2)) + 1, dtype='int')
# # The angle closest to -pi/2 is the azimuth, so the other
# # one is the back-azimuth
# tst1 = np.sum(Cxy2rza[bbf1:bbf2, smvc] * Cxy2[bbf1:bbf2, smvc], axis=0)/np.sum(Cxy2[bbf1:bbf2, smvc], axis=0) # noqa
# tst2 = np.sum(Cxy2rza2[bbf1:bbf2, smvc] * Cxy2[bbf1:bbf2, smvc], axis=0)/np.sum(Cxy2[bbf1:bbf2, smvc], axis=0) # noqa
# # See which one is the farthest from -pi/2
# bbvf = np.full(data.nits - nsmth, np.nan)
# for jj in range(0, len(bbv)):
#     tst_ind = np.argmax(np.abs(np.array([tst1[jj], tst2[jj]]) - (-np.pi/2)))
#     if tst_ind == 0:
#         bbvf[jj] = bbv[jj]
#     else:
#         bbvf[jj] = bbv2[jj]


#######################
# Plotting
#######################
fig, axs = plt.subplots(3, 1, sharex='col')
# Trace
axs[0].plot(tvec, data.Z, c='k')
axs[0].set_ylabel('Displacement \n [m]')
# Gamma^2
sc0 = axs[1].pcolormesh(t, freq, Cxy2, cmap=cc.cm.rainbow)
axs[1].axis('tight')
axs[1].set_xlim(t[0], t[-1])
axs[1].set_ylim(freq[0], freq[-1])
axs[1].set_ylabel('Frequency \n [Hz]', fontsize=12)
p1 = axs[1].get_position()
cbaxes1 = fig.add_axes([0.92, p1.y0, 0.02, p1.height])
hc1 = plt.colorbar(sc0, orientation="vertical",
                   cax=cbaxes1, ax=axs[1])
hc1.set_label('Max Weighted \n Coherence')
sc0.set_clim(0.0, 1.0)


# Back-azimuth Estimate
sc1 = axs[2].scatter(t[smvc], bbvf - 181, c=aa2, cmap=cc.cm.rainbow)
# axs[2].set_ylim(-180, 180)
axs[2].axhline(-52)
axs[2].set_ylim(-180, 180)
axs[2].set_ylabel('Back-Azimuth \n [Deg.]', fontsize=12)
p1 = axs[2].get_position()
cbaxes1 = fig.add_axes([0.92, p1.y0, 0.02, p1.height])
hc1 = plt.colorbar(sc1, orientation="vertical",
                   cax=cbaxes1, ax=axs[2])
hc1.set_label('Max Weighted \n Coherence')
sc1.set_clim(0.0, 1.0)

axs[2].xaxis_date()
axs[2].tick_params(axis='x', labelbottom='on')
axs[2].fmt_xdata = dates.DateFormatter('%HH:%MM')
axs[2].xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
axs[2].set_xlabel('UTC Time')
# fig.savefig('Python_TCM_Example.png', bbox_inches='tight', dpi=300, facecolor="w") # noqa

os.getcwd()
