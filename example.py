from obspy import read

from tcm.classes import tcm_classes, tcm_data_class
from tcm.tools import plotting

# Filter range [Hz]
freq_min = 10.0
freq_max = 20.0

# Window length [sec]
window_length = 8.0

# Fraction of window overlap [0.0, 1.0)
window_overlap = 0.90

# Azimuths to scans over [degrees]
az_min = 0.0
az_max = 359.0
az_delta = 1.0

################
# End User Input
################
# Read example data
# A signal from Great Sitkin that was recorded at GSMY
st = read('TCM_Example_GSMY.mseed')
st.sort(['channel'], reverse=True)
#st[0].plot()

# Create object to hold data and pre-process
data = tcm_data_class.DataBin(freq_min, freq_max,
                              window_length, window_overlap,
                              az_min, az_max, az_delta)
data.build_data_arrays(st)

# Create cross-spectral matrix object
CSM = tcm_classes.Spectral(data)
# Calculate spectra and cross-spectra
CSM.calculate_spectral_matrices(data)
# Calculate the vertical coherence
#CSM.calculate_vertical_Cxy2(data)
# Calculate the transverse coherence over all trial azimuths
CSM.calculate_tcm_over_azimuths(data)
# Find the coherence minima and apply the retrograde assumption if applicable
baz, sigma = CSM.find_minimum_tc(data)

sigma
# Plot the results
fig, axs = plotting.tcm_plot(data, CSM)
# fig.savefig('Python_TCM_Example.png', bbox_inches='tight', dpi=300, facecolor="w") # noqa

# fig.savefig('Python_TCM_Example_0722.png', bbox_inches='tight', dpi=300, facecolor="w") # noqa
