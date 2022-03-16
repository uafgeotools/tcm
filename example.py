import colorcet as cc
import matplotlib.pyplot as plt
from matplotlib import dates
from obspy import read

import os as os
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