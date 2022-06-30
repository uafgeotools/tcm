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

# Filter range [Hz]
freq_min = 10.0
freq_max = 20.0

# Window length [sec]
window_length = 6.0

# Fraction of window overlap [0.0, 1.0)
window_overlap = 0.50

# Azimuths to scans over [degrees]
az_min = 0.0
az_max = 359.0
az_delta = 1.0

# Use retrograde motion to determine back-azimuth angle
assume_retrograde = True

################
# End User Input
################
# Read example data
st = read('TCM_Example_GSMY.mseed')
st.sort(['channel'], reverse=True)
st[0].plot()

# Create object to hold data and pre-process
data = tcm_data_class.DataBin(freq_min, freq_max, window_length, window_overlap, az_min, az_max, az_delta, assume_retrograde)
data.build_data_arrays(st)

# Create cross-spectral matrix object
CSM = tcm_classes.Spectral(data)
# Calculate spectra and cross-spectra
CSM.calculate_spectral_matrices(data)
# Calculate the vertical coherence
CSM.calculate_vertical_Cxy2(data)
# Calculate the transverse coherence over all trial azimuths
CSM.calculate_tcm_over_azimuths(data)
# Find the coherence minima and apply the retrograde assumption if applicable
baz = CSM.find_minimum_tc(data)

# Plot the results
fig, axs = plotting.tcm_plot(data, CSM)
# fig.savefig('Python_TCM_Example.png', bbox_inches='tight', dpi=300, facecolor="w") # noqa

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
