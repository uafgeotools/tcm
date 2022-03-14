import numpy as np
from scipy.signal import csd, windows
from scipy.fft import rfftfreq


class DataBin:
    """ Data container for TCM processing"""

    def __init__(self, freq_min, freq_max, window_length,
                 window_overlap, az_min, az_max,
                 az_delta, assume_retrograde=True):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.window_length = window_length
        self.window_overlap = window_overlap
        self.az_min = az_min
        self.az_max = az_max
        self.az_delta = az_delta
        self.assume_retrograde = assume_retrograde

    def build_data_arrays(self, st):
        # Assumes all traces have the same sample rate and length
        self.sampling_rate = st[0].stats.sampling_rate
        self.winlensamp = int(self.window_length * self.sampling_rate) # noqa
        # Sample increment (delta_t)
        self.sampinc = int((1 - self.window_overlap) * self.winlensamp) + 1
        # Time intervals
        self.intervals = np.arange(0, len(st[0].data) - self.winlensamp, self.sampinc, dtype='int') # noqa
        self.nits = len(self.intervals)
        # Pull time from stream object
        self.tvec = st[0].times('matplotlib')
        # Assign the traces to individual arrays
        st.sort(keys=['channel'])
        if len(st) == 4:
            self.Infra = st[0].data
            self.E = st[1].data
            self.N = st[2].data
            self.Z = st[3].data
        else:
            pass
            # Raise an Error here for streams with more/less than 4 channels


class Spectral:
    """ A cross spectral matrix class"""

    def __init__(self, data):
        # Sub-window size
        self.sub_window = int(data.winlensamp/2)
        # FFT length (power of 2)
        self.nfft = np.power(2, int(np.ceil(np.log2(data.winlensamp))))
        # Filter for FFT
        self.window = windows.hamming(self.sub_window, sym=False)
        # FFT frequency vector
        self.freq_vector = rfftfreq(self.nfft, 1/data.sampling_rate)
        # Pre-allocate time vector
        self.t = np.full(data.nits, np.nan)
        # Pre-allocate cross spectral matrices
        # Vertical and Infrasound
        self.S_zi = np.empty((len(self.freq_vector),
                              data.nits), dtype=np.complex128)
        # Infrasound
        self.S_ii = np.full((len(self.freq_vector), data.nits), np.nan)
        # Vertical
        self.S_zz = np.full((len(self.freq_vector), data.nits), np.nan)
        # Squared Coherence
        self.C_xy2 = np.full((len(self.freq_vector), data.nits), np.nan)

    def calculate_vertical_Cxy2(self, data):
        # Loop through time and calculate the CSD at each frequency
        for jj in range(0, data.nits):
            # Get time from middle of window, except for the end.
            t0_ind = data.intervals[jj]
            tf_ind = data.intervals[jj] + data.winlensamp
            try:
                self.t[jj] = data.tvec[t0_ind + self.sub_window]
            except:
                self.t[jj] = np.nanmax(self.t, axis=0)

            _, self.S_zi[:, jj] = csd(
                data.Z[t0_ind:tf_ind], data.Infra[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=self.sub_window, noverlap=50, nfft=self.nfft)
            _, self.S_ii[:, jj] = np.real(csd(
                data.Infra[t0_ind:tf_ind], data.Infra[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=self.sub_window, noverlap=50, nfft=self.nfft))
            _, self.S_zz[:, jj] = np.real(csd(
                data.Z[t0_ind:tf_ind], data.Z[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=self.sub_window, noverlap=50, nfft=self.nfft))

            # Calculate the normalized coherence between the vertical
            # seismic channel and the infrasound channel
            self.Cxy2[:, jj] = np.real(np.multiply(self.S_zi[:, jj], np.conjugate(self.S_zi[:, jj]))) / np.multiply(self.S_ii[:, jj], self.S_zz[:, jj]) # noqa

    def build_azimuths(self, data):
        pass