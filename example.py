from obspy import read

from tcm.classes import tcm_classes, tcm_data_class
from tcm.tools import plotting

# from waveform_collection import gather_waveforms

# Read example data
st = read('TCM_Example_GSMY.mseed')

# Filter range [Hz]
freq_min = 10.0
freq_max = 20.0

# Window length [sec]
window_length = 4.0

# Fraction of window overlap [0.0, 1.0)
window_overlap = 0.90

# Azimuths to scans over [degrees]
az_min = -179
az_max = 180
az_delta = 1

# Use retrograde motion to determine back-azimuth angle
assume_retrograde = True

################
# End User Input
################

# import time
# t = time.time()

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

# elapsed = time.time() - t
# print(elapsed)

# Plot the results
fig, axs = plotting.tcm_plot(data, CSM)
