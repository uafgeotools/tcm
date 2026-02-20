from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from tcm import tcm
from tcm.tools import plotting
from matplotlib import rcParams
rcParams.update({'font.size': 10})
#%%

# Filter range [Hz]
freq_min = 10.0
freq_max = 20.0

# Use 2 Hz narrowband [True] or broadband [False] coherence maxima for calculation
search_2Hz = False

# Window length [sec]
window_length = 15.0

# Fraction of window overlap [0.0, 1.0)
window_overlap = 0.90

# Azimuths to scans over [degrees]
az_min = 0.0
az_max = 359.0
az_delta = 1.0

''' Our example signal is an eruption from Great Sitkin Volcano, Alaska
    that was recorded by AVO station GSMY. '''
STARTTIME = UTCDateTime('2021-5-26T05:03:00')
ENDTIME = STARTTIME + 3*60

# Station information, including fdsn/etc client to read from
SOURCE = 'IRIS'
NETWORK = 'AV'
STATION = 'GSMY'
LOCATION = '*'
CHANNEL = 'BDF,BHZ,BHN,BHE'

#%% Read in data and pre-process
client = Client(SOURCE)
st = client.get_waveforms(NETWORK, STATION, LOCATION, CHANNEL,
                          starttime=STARTTIME, endtime=ENDTIME,
                          attach_response='True')

# Remove response
for tr in st:
    fs_resp = tr.stats.sampling_rate
    # Pre-filt for response removal
    pre_filt = [0.0005, 0.001, fs_resp/2-2, fs_resp/2]
    if tr.stats.channel[1:] == 'DF':
        tr.remove_response(pre_filt=pre_filt, output='VEL', water_level=None)
    else:
        tr.remove_response(pre_filt=pre_filt, output='DISP', water_level=None)

# Sort by component: E, F, N, Z
st.sort(keys=['component'])

# Ensure sample rate consistency
st.interpolate(sampling_rate=st[0].stats.sampling_rate, method='lanczos', a=15)
st.detrend(type='linear')

#%% Run the transverse coherence minimization (TCM) algorithm
baz, sigma, time_smooth, frequency_vector, time, Cxy2, median_coherence, freq_lim_min, freq_lim_max = tcm.run_tcm(st, freq_min, freq_max, window_length, window_overlap, az_min, az_max, az_delta, search_2Hz) # noqa

#%% Plot the results
fig, axs = plotting.tcm_plot(st, freq_min, freq_max, baz,
                             time_smooth, frequency_vector, time,
                             Cxy2, median_coherence, freq_lim_min, freq_lim_max,
                             search_2Hz)
# Plot uncertainties
axs[4].scatter(time_smooth, baz + sigma, c='gray', marker='_', linestyle=':')
axs[4].scatter(time_smooth, baz - sigma, c='gray', marker='_', linestyle=':')\
fig.show()
