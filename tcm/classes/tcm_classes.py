import numpy as np
from scipy.signal import csd, windows
from scipy.fft import rfftfreq
from numba import jit


@jit(nopython=True)
def _calculate_vertical_Cxy2(Cxy2, S_zi, S_ii, S_zz, nits):
    """ Calculate the vertical magnitude-squared coherence """
    # Loop through time and calculate the CSD at each frequency
    for jj in range(0, nits):
        """Calculate the normalized coherence between the vertical
            seismic channel and the infrasound channel """
        Cxy2[:, jj] = np.real(np.multiply(S_zi[:, jj], np.conjugate(S_zi[:, jj]))) / np.multiply(S_ii[:, jj], S_zz[:, jj]) # noqa

    return Cxy2


@jit(nopython=True)
def _calculate_tcm_over_azimuths(nits, az_vector, Cxy2, Cxy2_trial, S_ni,
                                 S_ei, S_nn, S_ne, S_ee, S_ii,
                                 weighted_coherence_v, sum_coherence_v,
                                 fmin_ind, fmax_ind):
    # Loop over time
    for jj in range(0, nits):
        """ Loop over all azimuths and calculate transverse
        coherence for the trial azimuth. """
        for kk in range(0, len(az_vector)):
            Cxy2_trial[:, jj] = (np.abs(S_ni[:, jj] * np.sin(az_vector[kk] * np.pi/180) - S_ei[:, jj] * np.cos(az_vector[kk] * np.pi/180))**2) / (np.abs(S_nn[:, jj] * np.sin(az_vector[kk] * np.pi/180)**2 - np.real(S_ne[:, jj]) * np.sin(2 * az_vector[kk] * np.pi/180) + S_ee[:, jj] * np.cos(az_vector[kk] * np.pi/180)**2) * np.abs(S_ii[:, jj])) # noqa

            """ Weighting trial transverse coherence values using
                the vertical coherence. """
            weighted_coherence_v[kk, jj] = np.sum(Cxy2_trial[fmin_ind:fmax_ind, jj] * Cxy2[fmin_ind:fmax_ind, jj]) # noqa
            # Sum of vertical coherence for denominator of weighted sum
            sum_coherence_v[kk, jj] = np.sum(Cxy2[fmin_ind:fmax_ind, jj])

    weighted_coherence = weighted_coherence_v/sum_coherence_v # noqa

    return weighted_coherence


def calc_csm(x1, x2, dt, window=None, sub_window_len=None, sub_window_overlap=0.5, fft_window="hanning", normalize_windowing=False):
    """Compute the Fourier transform of the array data to perform analysis

        Compute the Fourier transform of the array data within an analysis window defined by window = [t1, t2]
        and potentially using subwindows to obtain a full rank covariance matrix for beamforming analyses
        requiring such data.  Multiple FFT window options are available and a normalization option scales to
        account for the amplitude loss at the window edges.

        Parameters
        ----------
        x : 2darray
            M x N matrix of array data, x[m][n] = x_m(t_n)
        t : 1darray
            Vector of N sampled points in time, t[n] = t_n
        window : float
            Start and end time of the window relative to times in t, [t_1, t_2]
        sub_window_len : float
            Duration of the subwindow in seconds
        sub_window_overlap : float
            Fraction of subwindow to overlap (limited range of 0.0 to 0.9)
        fft_window : str
            Fourier windowing method
        normalize_windowing : boolean
            Boolean to apply normalization of window scaling

        Returns:
        ----------
        f : 1darray
            Vector of N_f frequencies for the FFT'd data, f[n] = f_n
        S : 3darray
            M x M x N_f cube of the covariance matrices, S[m1][m2][n] = mean(X_{m1}(f_n) conj(X_{m2}(f_n))
    """

    M = 1
    N = len(x1)
    x1 = x1.reshape(1, N)
    x2 = x2.reshape(1, N)

    win_n1, win_N = 0, N

    padded_N = 2**int(np.ceil(np.log2(win_N)))
    N_f = int(padded_N / 2 + 1)

    f = (1.0 / dt) * (np.arange(float(N_f)) / padded_N)
    X1 = np.zeros((M, N_f), dtype=complex)
    X2 = np.zeros((M, N_f), dtype=complex)
    S = np.zeros((M, M, N_f), dtype=complex)

    # window and zero pad the data
    temp1 = np.zeros((M, padded_N))
    temp2 = np.zeros((M, padded_N))
    temp1[:, 0:win_N] = x1[:, win_n1:win_n1 + win_N]
    temp2[:, 0:win_N] = x2[:, win_n1:win_n1 + win_N]
    if fft_window == "hanning":
        temp1[:, 0:win_N] *= np.array([np.hanning(win_N)] * M)
        temp2[:, 0:win_N] *= np.array([np.hanning(win_N)] * M)
        if normalize_windowing:
            temp1 /= np.mean(np.hanning(win_N))
            temp2 /= np.mean(np.hanning(win_N))
    elif fft_window == "bartlett":
        temp1[:, 0:win_N] *= np.array([np.bartlett(win_N)] * M)
        temp2[:, 0:win_N] *= np.array([np.bartlett(win_N)] * M)
        if normalize_windowing:
            temp1 /= np.mean(np.bartlett(win_N))
            temp2 /= np.mean(np.bartlett(win_N))
    elif fft_window == "blackman":
        temp1[:, 0:win_N] *= np.array([np.blackman(win_N)] * M)
        temp2[:, 0:win_N] *= np.array([np.blackman(win_N)] * M)
        if normalize_windowing:
            temp1 /= np.mean(np.blackman(win_N))
            temp2 /= np.mean(np.blackman(win_N))
    elif fft_window == "hamming":
        temp1[:, 0:win_N] *= np.array([np.hamming(win_N)] * M)
        temp2[:, 0:win_N] *= np.array([np.hamming(win_N)] * M)
        if normalize_windowing:
            temp1 /= np.mean(np.hamming(win_N))
            temp2 /= np.mean(np.hamming(win_N))
    else:
        msg = "Unrecognized method in fft_window.  Options are 'hanning', 'bartlett', 'blackman', 'hamming', or 'boxcar'."
        temp1[:, 0:win_N] *= 1.0
        raise ValueError(msg)

    # fft the data and define X(f) and S(f)
    X1 = np.fft.rfft(temp1, axis=1) * dt
    X2 = np.fft.rfft(temp2, axis=1) * dt
    for nf in range(0, int(padded_N / 2) + 1):
        S[:, :, nf] = np.outer(X1[:, nf], np.conj(X2[:, nf]))

    return f, S.flatten()


class Spectral:
    """ A cross spectral matrix class"""

    def __init__(self, data):
        """ Pre-allocate arrays and assignment of FFT-related variables. """
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
        # Create azimuth vector [degrees]
        self.az_vector = np.array(np.arange(data.az_min, data.az_max + data.az_delta, data.az_delta)) # noqa
        # Get the closest frequency points to the preferred ones
        self.fmin_ind = np.argmin(np.abs(data.freq_min - self.freq_vector))
        self.fmax_ind = np.argmin(np.abs(data.freq_max - self.freq_vector))
        # Pre-allocate cross spectral matrices (S)
        # Vertical and Infrasound
        self.S_zi = np.empty((len(self.freq_vector),
                              data.nits), dtype=np.complex)
        # Infrasound
        self.S_ii = np.full((len(self.freq_vector), data.nits), np.nan)
        # Vertical
        self.S_zz = np.full((len(self.freq_vector), data.nits), np.nan)
        # East-Infrasound
        self.S_ei = np.empty((len(self.freq_vector), data.nits), dtype=np.complex) # noqa
        # North-Infrasound
        self.S_ni = np.empty((len(self.freq_vector), data.nits), dtype=np.complex) # noqa
        # East-East
        self.S_ee = np.empty((len(self.freq_vector), data.nits), dtype=np.complex) # noqa
        # North-North
        self.S_nn = np.empty((len(self.freq_vector), data.nits), dtype=np.complex) # noqa
        # North-East
        self.S_ne = np.empty((len(self.freq_vector), data.nits), dtype=np.complex) # noqa
        # Magnitude Squared Coherence
        self.Cxy2 = np.full((len(self.freq_vector), data.nits), np.nan)
        # Pre-allocate trial azimuth transverse-coherence matrices
        self.Cxy2_trial = np.empty((len(self.freq_vector), data.nits))
        self.weighted_coherence_v = np.empty((len(self.az_vector), data.nits))
        self.sum_coherence_v = np.empty((len(self.az_vector), data.nits))
        self.weighted_coherence = np.empty((len(self.freq_vector), data.nits))

    def calculate_spectral_matrices(self, data):
        """ Calculate the cross spectral matrices """

        """ Loop over time windows and calculate the cross spectrum
            at every frequency. """
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
                nperseg=None,
                noverlap=None, nfft=self.nfft)
            _, self.S_ii[:, jj] = np.real(csd(
                data.Infra[t0_ind:tf_ind], data.Infra[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=None,
                noverlap=None, nfft=self.nfft))
            _, self.S_zz[:, jj] = np.real(csd(
                data.Z[t0_ind:tf_ind], data.Z[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=None,
                noverlap=None, nfft=self.nfft))
            _, self.S_ei[:, jj] = csd(
                data.E[t0_ind:tf_ind], data.Infra[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=None,
                noverlap=None, nfft=self.nfft)
            _, self.S_ni[:, jj] = csd(
                data.N[t0_ind:tf_ind], data.Infra[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=None,
                noverlap=None, nfft=self.nfft)
            _, self.S_ee[:, jj] = np.real(csd(
                data.E[t0_ind:tf_ind], data.E[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=None,
                noverlap=None, nfft=self.nfft))
            _, self.S_nn[:, jj] = np.real(csd(
                data.N[t0_ind:tf_ind], data.N[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=None,
                noverlap=None, nfft=self.nfft))
            _, self.S_ne[:, jj] = csd(
                data.N[t0_ind:tf_ind], data.E[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=None,
                noverlap=None, nfft=self.nfft)

    def calculate_vertical_Cxy2(self, data):
        """ Calculate the vertical magnitude-squared coherence """
        self.Cxy2 = _calculate_vertical_Cxy2(self.Cxy2, self.S_zi, self.S_ii, self.S_zz, data.nits) # noqa

    def calculate_tcm_over_azimuths(self, data):
        self.weighted_coherence = _calculate_tcm_over_azimuths(data.nits, self.az_vector, self.Cxy2, self.Cxy2_trial, self.S_ni, self.S_ei, self.S_nn, self.S_ne, self.S_ee, self.S_ii, self.weighted_coherence_v, self.sum_coherence_v, self.fmin_ind, self.fmax_ind) # noqa

    def find_minimum_tc(self, data):
        # Apply some smoothing if desired
        # Number of samples in coherogram to smooth
        self.nsmth = 4
        self.bbv = np.full(data.nits - self.nsmth, 0, dtype='int')
        self.bbv2 = np.full(data.nits - self.nsmth, 0, dtype='int')
        self.aa2 = np.full(data.nits - self.nsmth, np.nan)
        self.bb2 = np.full(data.nits - self.nsmth, 0, dtype='int')

        # Here are the 2 possible back-azimuths
        for jj in range(0, data.nits - self.nsmth):
            idx = np.argsort(np.sum(
                self.weighted_coherence[:, jj:(jj + self.nsmth + 1)], 1))
            self.bbv[jj] = idx[0]
            self.bbv2[jj] = idx[1]
            # Info on the amount of coherence
            self.aa2[jj] = np.max(np.mean(
                self.weighted_coherence[:, jj:(jj + self.nsmth + 1)], 1))
            self.bb2[jj] = np.argmax(np.mean(
                self.weighted_coherence[:, jj:(jj + self.nsmth + 1)], 1))

        # Resolve the 180 degree ambiguity
        # Make this a flag that defaults to `True`
        self.Cxy2rz = np.empty((len(self.freq_vector), data.nits), dtype=np.complex) # noqa
        self.Cxy2rz2 = np.empty((len(self.freq_vector), data.nits), dtype=np.complex) # noqa
        self.Cxy2rza = np.empty((len(self.freq_vector), data.nits)) # noqa
        self.Cxy2rza2 = np.empty((len(self.freq_vector), data.nits)) # noqa
        for jj in range(0, data.nits - self.nsmth):
            t0_ind = data.intervals[jj]
            tf_ind = data.intervals[jj] + data.winlensamp
            _, self.Cxy2rz[:, jj] = csd(
                data.Z[t0_ind:tf_ind], data.N[t0_ind:tf_ind] * np.cos(
                    self.az_vector[self.bbv[jj]] * np.pi/180) + data.E[t0_ind:tf_ind] * np.sin(
                        self.az_vector[self.bbv[jj]] * np.pi/180), fs=data.sampling_rate, window=self.window,
                nperseg=None,
                noverlap=None, nfft=self.nfft) # noqa
            _, self.Cxy2rz2[:, jj] = csd(
                data.Z[t0_ind:tf_ind], data.N[t0_ind:tf_ind] * np.cos(
                    self.az_vector[self.bbv2[jj]] * np.pi/180) + data.E[t0_ind:tf_ind] * np.sin(self.az_vector[self.bbv2[jj]] * np.pi/180), fs=data.sampling_rate, window=self.window,
                nperseg=None,
                noverlap=None, nfft=self.nfft) # noqa
            self.Cxy2rza[:, jj] = np.angle(self.Cxy2rz[:, jj])
            self.Cxy2rza2[:, jj] = np.angle(self.Cxy2rz2[:, jj])
        # The time vector for the case of nonzero smoothing
        self.smvc = np.arange(((self.nsmth/2) + 1), (data.nits - (self.nsmth/2)) + 1, dtype='int') # noqa
        # The angle closest to -pi/2 is the azimuth, so the other
        # one is the back-azimuth
        tst1 = np.sum(self.Cxy2rza[self.fmin_ind:self.fmax_ind, self.smvc] * self.Cxy2[self.fmin_ind:self.fmax_ind, self.smvc], axis=0)/np.sum(self.Cxy2[self.fmin_ind:self.fmax_ind, self.smvc], axis=0) # noqa
        tst2 = np.sum(self.Cxy2rza2[self.fmin_ind:self.fmax_ind, self.smvc] * self.Cxy2[self.fmin_ind:self.fmax_ind, self.smvc], axis=0)/np.sum(self.Cxy2[self.fmin_ind:self.fmax_ind, self.smvc], axis=0) # noqa
        # See which one is the farthest from -pi/2
        self.baz_final = np.full(data.nits - self.nsmth, np.nan)
        for jj in range(0, len(self.bbv)):
            tst_ind = np.argmax(np.abs(np.array([tst1[jj], tst2[jj]]) - (-np.pi/2))) # noqa
            if tst_ind == 1:
                self.baz_final[jj] = self.az_vector[self.bbv[jj]]
            else:
                self.baz_final[jj] = self.az_vector[self.bbv2[jj]]

        # Convert azimuth to back-azimuth
        # self.baz_final -= 181

        return self.baz_final
