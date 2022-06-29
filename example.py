from obspy import read

from tcm.classes import tcm_classes, tcm_data_class
from tcm.tools import plotting

import matplotlib.pyplot as plt
from matplotlib import dates
from scipy.signal import csd
import colorcet as cc
import numpy as np

# from waveform_collection import gather_waveforms

import os as os
os.chdir('/Users/jwbishop/Documents/Github/tcm')

# Read example data
st = read('TCM_Example_GSMY.mseed')

# Filter range [Hz]
freq_min = 10.0
freq_max = 20.0

# Window length [sec]
window_length = 6.0

# Fraction of window overlap [0.0, 1.0)
window_overlap = 0.50

# Azimuths to scans over [degrees]
az_min = -179
az_max = 180
az_delta = 1.0

# Use retrograde motion to determine back-azimuth angle
assume_retrograde = True

################
# End User Input
################

# from waveform_collection import gather_waveforms
# from obspy.core import UTCDateTime
# NET = 'AV'
# STA = 'RDT'
# CHAN = '*'
# LOC = '*'
# START = UTCDateTime('2022-01-15T13:00')
# END = START + 60*60
# st = gather_waveforms('IRIS', NET, STA, LOC, CHAN, START, END, remove_response=False)

# st.filter('bandpass', freqmin=freq_min, freqmax=freq_max, corners=2, zerophase=True)
# st.taper(max_percentage=0.01)
st.sort(['channel'], reverse=True)
st[0].plot()

# Create object to hold data and pre-process
data = tcm_data_class.DataBin(freq_min, freq_max, window_length, window_overlap, az_min, az_max, az_delta, assume_retrograde)
data.build_data_arrays(st)

# Create cross-spectral matrix object
CSM = tcm_classes.Spectral(data)
# Calculate spectra and cross-spectra

import time as time
a = time.time()
b = time.time()
# results = Parallel(n_jobs=-1)(delayed(narrow_band_loop)(ii, freqlist, FREQ_BAND_TYPE, freq_resp_list, st, FILTER_TYPE, FILTER_ORDER, FILTER_RIPPLE, rij, WINLEN_list, WINOVER, ALPHA, vector_len) for ii in range(NBANDS))

import multitaper.mtcross as cross
def MTC(jj, data_intervals, data_winlensamp, data_Z, data_Infra, data_N, data_E, nw, kspec, dt, nfft, freq_vector):
    t0_ind = data_intervals[jj]
    tf_ind = data_intervals[jj] + data.winlensamp
    SSz = cross.MTCross(data_Z[t0_ind:tf_ind], data_Infra[t0_ind:tf_ind], nw, kspec, dt, nfft=nfft)
    Szi = SSz.Sxy[0:len(freq_vector)].flatten()
    Szz = SSz.Sxx[0:len(freq_vector)].flatten()
    Sii = SSz.Syy[0:len(freq_vector)].flatten()
    SSe = cross.MTCross(data_E[t0_ind:tf_ind], data_Infra[t0_ind:tf_ind], nw, kspec, dt, nfft=nfft)
    See = SSe.Sxx[0:len(freq_vector)].flatten()
    Sei = SSe.Sxy[0:len(freq_vector)].flatten()
    SSn = cross.MTCross(data_N[t0_ind:tf_ind], data_Infra[t0_ind:tf_ind], nw, kspec, dt, nfft=nfft)
    Snn = SSn.Sxx[0:len(freq_vector)].flatten()
    Sni = SSn.Sxy[0:len(freq_vector)].flatten()
    SSne = cross.MTCross(data_N[t0_ind:tf_ind], data_E[t0_ind:tf_ind], nw, kspec, dt, nfft=nfft)
    Sne = SSne.Sxy[0:len(freq_vector)].flatten()
    return Szi, Szz, Sii, See, Sei, Snn, Sni, Sne
    # return Szi, Szz, Sii

nw = 7.5
kspec = 10
from joblib import Parallel, delayed
a = time.time()
results = Parallel(n_jobs=-1)(delayed(MTC)(jj, data.intervals, data.winlensamp, data.Z, data.Infra, data.N, data.E, 7.5, 10, 1/data.sampling_rate, CSM.nfft, CSM.freq_vector) for jj in range(0, data.nits))
b = time.time()
print(b-a)

results = np.array(results, dtype=np.complex)

CSM.S_zi = results[:, 0, :].T
CSM.S_zz = results[:, 1, :].T
CSM.S_ii = results[:, 2, :].T
CSM.S_ee = results[:, 3, :].T
CSM.S_ei = results[:, 4, :].T
CSM.S_nn = results[:, 5, :].T
CSM.S_ni = results[:, 6, :].T
CSM.S_ne = results[:, 7, :].T


# for jj in range(0, data.nits):
#     print(jj/(data.nits) * 100)
#     # Get time from middle of window, except for the end.
#     t0_ind = data.intervals[jj]
#     tf_ind = data.intervals[jj] + data.winlensamp
#     try:
#         CSM.t[jj] = data.tvec[t0_ind + CSM.sub_window]
#     except:
#         CSM.t[jj] = np.nanmax(CSM.t, axis=0)
#
#     Sz = cross.MTCross(data.Z[t0_ind:tf_ind], data.Infra[t0_ind:tf_ind], nw, kspec, dt, nfft=CSM.nfft)
#     CSM.S_zi[:, jj] = Sz.Sxy[0:len(CSM.freq_vector)].flatten()
#     CSM.S_zz[:, jj] = Sz.Sxx[0:len(CSM.freq_vector)].flatten()
#     CSM.S_ii[:, jj] = Sz.Syy[0:len(CSM.freq_vector)].flatten()
#     CSM.Cxy2[:, jj] = Sz.cohe[0:len(CSM.freq_vector)].flatten()
#     Se = cross.MTCross(data.E[t0_ind:tf_ind], data.Infra[t0_ind:tf_ind], nw, kspec, dt, nfft=CSM.nfft)
#     CSM.S_ee[:, jj] = Se.Sxx[0:len(CSM.freq_vector)].flatten()
#     CSM.S_ei[:, jj] = Se.Sxy[0:len(CSM.freq_vector)].flatten()
#     Sn = cross.MTCross(data.N[t0_ind:tf_ind], data.Infra[t0_ind:tf_ind], nw, kspec, dt, nfft=CSM.nfft)
#     CSM.S_nn[:, jj] = Sn.Sxx[0:len(CSM.freq_vector)].flatten()
#     CSM.S_ni[:, jj] = Sn.Sxy[0:len(CSM.freq_vector)].flatten()
#     Sne = cross.MTCross(data.N[t0_ind:tf_ind], data.E[t0_ind:tf_ind], nw, kspec, dt, nfft=CSM.nfft)
#     CSM.S_ne[:, jj] = Sne.Sxy[0:len(CSM.freq_vector)].flatten()
#
#
# CSM.S_ii = np.array(CSM.S_ii, dtype=np.complex)
# CSM.S_zz = np.array(CSM.S_zz, dtype=np.complex)
# CSM.S_zi = np.array(CSM.S_zi, dtype=np.complex)
# CSM.S_nn = np.array(CSM.S_nn, dtype=np.complex)
# CSM.S_ee = np.array(CSM.S_ee, dtype=np.complex)
# CSM.S_ei = np.array(CSM.S_ei, dtype=np.complex)
# CSM.S_ne = np.array(CSM.S_ne, dtype=np.complex)
# CSM.S_ni = np.array(CSM.S_ni, dtype=np.complex)

CSM = tcm_classes.Spectral(data)
CSM.calculate_spectral_matrices(data)
CSM.S_zi = results[:, 0, :].T
CSM.S_zz = results[:, 1, :].T
CSM.S_ii = results[:, 2, :].T
# Calculate the vertical coherence
CSM.calculate_vertical_Cxy2(data)
# Calculate the transverse coherence over all trial azimuths
CSM.calculate_tcm_over_azimuths(data)
# Find the coherence minima and apply the retrograde assumption if applicable
baz = CSM.find_minimum_tc(data)


# Plot the results
fig, axs = plotting.tcm_plot(data, CSM)
# fig.savefig('Python_TCM_Example.png', bbox_inches='tight', dpi=300, facecolor="w") # noqa

# Make this a flag that defaults to `True`
CSM.t = np.full(data.nits, np.nan)
for jj in range(0, data.nits):
    # Get time from middle of window, except for the end.
    t0_ind = data.intervals[jj]
    tf_ind = data.intervals[jj] + data.winlensamp
    try:
        CSM.t[jj] = data.tvec[t0_ind + CSM.sub_window]
    except:
        CSM.t[jj] = np.nanmax(CSM.t, axis=0)

# Apply some smoothing if desired
CSM.bbv = np.full(data.nits - CSM.nsmth, 0, dtype='int')
CSM.bbv2 = np.full(data.nits - CSM.nsmth, 0, dtype='int')
CSM.aa2 = np.full(data.nits - CSM.nsmth, np.nan)
CSM.bb2 = np.full(data.nits - CSM.nsmth, 0, dtype='int')

# Here are the 2 possible back-azimuths
for jj in range(0, data.nits - CSM.nsmth):
    idx = np.argsort(np.sum(
        CSM.weighted_coherence[:, jj:(jj + CSM.nsmth + 1)], 1))
    CSM.bbv[jj] = idx[0]
    CSM.bbv2[jj] = idx[1]
    # Info on the amount of coherence
    CSM.aa2[jj] = np.max(np.mean(
        CSM.weighted_coherence[:, jj:(jj + CSM.nsmth + 1)], 1))
    CSM.bb2[jj] = np.argmax(np.mean(
        CSM.weighted_coherence[:, jj:(jj + CSM.nsmth + 1)], 1))

Cxy2rz = np.empty((len(CSM.freq_vector), data.nits))
Cxy2rz2 = np.empty((len(CSM.freq_vector), data.nits))
Cxy2rza = np.empty((len(CSM.freq_vector), data.nits))
Cxy2rza2 = np.empty((len(CSM.freq_vector), data.nits))
dt = 1/data.sampling_rate
for jj in range(0, data.nits - CSM.nsmth):
    print(jj/(data.nits - CSM.nsmth) * 100)
    t0_ind = data.intervals[jj]
    tf_ind = data.intervals[jj] + data.winlensamp
    y1 = data.N[t0_ind:tf_ind] * np.cos(CSM.az_vector[CSM.bbv[jj]] * np.pi/180) + data.E[t0_ind:tf_ind] * np.sin(CSM.az_vector[CSM.bbv[jj]] * np.pi/180)
    S = cross.MTCross(data.Z[t0_ind:tf_ind], y1, nw, kspec, dt, nfft=CSM.nfft)
    y2 = data.N[t0_ind:tf_ind] * np.cos(CSM.az_vector[CSM.bbv2[jj]] * np.pi/180) + data.E[t0_ind:tf_ind] * np.sin(CSM.az_vector[CSM.bbv2[jj]] * np.pi/180)
    S2 = cross.MTCross(data.Z[t0_ind:tf_ind], y2, nw, kspec, dt, nfft=CSM.nfft)
    Cxy2rz[:, jj] = S.cohe[0:len(CSM.freq_vector)].flatten()
    Cxy2rz2[:, jj] = S2.cohe[0:len(CSM.freq_vector)].flatten()
    Cxy2rza[:, jj] = S.phase[0:len(CSM.freq_vector)].flatten() * np.pi/180
    Cxy2rza2[:, jj] = S2.phase[0:len(CSM.freq_vector)].flatten() * np.pi/180
# The time vector for the case of nonzero smoothing
CSM.smvc = np.arange(((CSM.nsmth/2) + 1), (data.nits - (CSM.nsmth/2)) + 1, dtype='int') # noqa
# The angle closest to -pi/2 is the azimuth, so the other
# one is the back-azimuth
tst1 = np.sum(Cxy2rza[CSM.fmin_ind:CSM.fmax_ind, CSM.smvc] * CSM.Cxy2[CSM.fmin_ind:CSM.fmax_ind, CSM.smvc], axis=0)/np.sum(CSM.Cxy2[CSM.fmin_ind:CSM.fmax_ind, CSM.smvc], axis=0) # noqa
tst2 = np.sum(Cxy2rza2[CSM.fmin_ind:CSM.fmax_ind, CSM.smvc] * CSM.Cxy2[CSM.fmin_ind:CSM.fmax_ind, CSM.smvc], axis=0)/np.sum(CSM.Cxy2[CSM.fmin_ind:CSM.fmax_ind, CSM.smvc], axis=0) # noqa
# Pick the angle the farthest from -pi/2
CSM.baz_final = np.full(data.nits - CSM.nsmth, np.nan)
for jj in range(0, len(CSM.bbv)):
    tst_ind = np.argmax(np.abs(np.array([tst1[jj], tst2[jj]]) - (-np.pi/2))) # noqa
    if tst_ind == 0:
        CSM.baz_final[jj] = CSM.az_vector[CSM.bbv[jj]]
    else:
        CSM.baz_final[jj] = CSM.az_vector[CSM.bbv2[jj]]


CSM.baz_final = (CSM.baz_final + 360) % 360

fig, axs = plt.subplots(3, 1, sharex='col', figsize=(8, 11))
# Vertical component of seismic trace
axs[0].plot(data.tvec, data.Z, c='k')
axs[0].set_ylabel('Displacement \n [m]')
# Magnitude squared coherence
# sc0 = axs[1].pcolormesh(CSM.t, CSM.freq_vector, CSM.Cxy2,
#                         cmap='plasma', shading='auto')
sc0 = axs[1].pcolormesh(CSM.t, CSM.freq_vector, CSM.Cxy2,
                            cmap=cc.cm.rainbow, shading='auto')
axs[1].axis('tight')
# axs[1].set_xlim(CSM.t[0], CSM.t[-1])
axs[1].set_ylim(data.freq_min, data.freq_max)
axs[1].set_ylabel('Frequency \n [Hz]', fontsize=12)
p1 = axs[1].get_position()
cbaxes1 = fig.add_axes([0.92, p1.y0, 0.02, p1.height])
hc1 = plt.colorbar(sc0, orientation="vertical",
                   cax=cbaxes1, ax=axs[1])
hc1.set_label('Max Weighted \n Coherence')
sc0.set_clim(0.0, 1.0)

# Back-azimuth Estimate
sc1 = axs[2].scatter(CSM.t[CSM.smvc], CSM.baz_final, 12,
                     c=CSM.aa2, cmap=cc.cm.rainbow)
# sc1 = axs[2].scatter(CSM.t[CSM.smvc], CSM.baz_final, c=CSM.aa2, cmap='plasma')
# axs[2].set_ylim(-180, 180)
axs[2].axhline(-52+360, c='k', linestyle=':')
axs[2].set_ylim(0, 360)
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


CSM.S_nn
fig, ax = plt.subplots(1)
hand = ax.pcolormesh(np.real(CSM.S_ne[CSM.fmin_ind:CSM.fmax_ind, : ]))
plt.colorbar(hand)

CSM2 = tcm_classes.Spectral(data)
CSM2.calculate_spectral_matrices(data)
CSM2.calculate_vertical_Cxy2(data)
CSM2.calculate_tcm_over_azimuths(data)
baz = CSM2.find_minimum_tc(data)
fig, axs = plotting.tcm_plot(data, CSM2)


Cxy2_trial = np.empty((len(CSM.freq_vector), data.nits))
weighted_coherence_v = np.empty((len(CSM.az_vector), data.nits))
sum_coherence_v = np.empty((len(CSM.az_vector), data.nits))
weighted_coherence = np.empty((len(CSM.freq_vector), data.nits))

for jj in range(0, data.nits):
    """ Loop over all azimuths and calculate transverse
    coherence for the trial azimuth. """
    for kk in range(0, len(CSM.az_vector)):
        Cxy2_trial[:, jj] = (np.abs(CSM.S_ni[:, jj] * np.sin(CSM.az_vector[kk] * np.pi/180) - CSM.S_ei[:, jj] * np.cos(CSM.az_vector[kk] * np.pi/180))**2) / (np.abs(CSM.S_nn[:, jj] * np.sin(CSM.az_vector[kk] * np.pi/180)**2 - np.real(CSM.S_ne[:, jj]) * np.sin(2 * CSM.az_vector[kk] * np.pi/180) + CSM.S_ee[:, jj] * np.cos(CSM.az_vector[kk] * np.pi/180)**2) * np.abs(CSM.S_ii[:, jj])) # noqa

        """ Weighting trial transverse coherence values using
            the vertical coherence. """
        weighted_coherence_v[kk, jj] = np.sum(Cxy2_trial[CSM.fmin_ind:CSM.fmax_ind, jj] * CSM.Cxy2[CSM.fmin_ind:CSM.fmax_ind, jj]) # noqa
        # Sum of vertical coherence for denominator of weighted sum
        sum_coherence_v[kk, jj] = np.sum(CSM.Cxy2[CSM.fmin_ind:CSM.fmax_ind, jj])

weighted_coherence = weighted_coherence_v/sum_coherence_v

