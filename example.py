from obspy import read

from tcm import tcm
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

# Run the transverse coherence minimization algorithm
baz, sigma, time_smooth, frequency_vector, time, Cxy2, mean_coherence = tcm.run_tcm(st, freq_min, freq_max, window_length, window_overlap, az_min, az_max, az_delta) # noqa

# Plot the results
fig, axs = plotting.tcm_plot(st, freq_min, freq_max, baz,
                             time_smooth, frequency_vector, time,
                             Cxy2, mean_coherence)
# Plot uncertainties
axs[2].scatter(time_smooth, baz + sigma, c='gray', marker='_', linestyle=':')
axs[2].scatter(time_smooth, baz - sigma, c='gray', marker='_', linestyle=':')
# fig.savefig('Python_TCM_Example.png', bbox_inches='tight', dpi=300, facecolor="w") # noqa
