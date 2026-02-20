import numpy as np
from scipy.signal import csd, coherence
from scipy.fft import rfftfreq
from numba import jit


@jit(nopython=True)
def _calculate_tcm_over_azimuths(nits, az_vector, Cxy2, Cxy2_trial, S_ni,
                                 S_ei, S_nn, S_ne, S_ee, S_ii,
                                 ZI_coherence, TI_coherence,
                                 fmin_ind, fmax_ind):
    # Loop over time
    for jj in range(0, nits):
        """ Loop over all azimuths and calculate transverse
        coherence for the trial azimuth. """
        for kk in range(0, len(az_vector)):
            Cxy2_trial[:, jj] = (np.abs(S_ni[:, jj] * np.sin(az_vector[kk] * np.pi/180) - S_ei[:, jj] * np.cos(az_vector[kk] * np.pi/180))**2) / (np.abs(S_nn[:, jj] * np.sin(az_vector[kk] * np.pi/180)**2 - np.real(S_ne[:, jj]) * np.sin(2 * az_vector[kk] * np.pi/180) + S_ee[:, jj] * np.cos(az_vector[kk] * np.pi/180)**2) * np.abs(S_ii[:, jj])) # noqa

            """ Unweighted, purely T-I coherence minimization. """
            ZI_coherence[kk, jj] = np.median(Cxy2[fmin_ind[jj]:fmax_ind[jj], jj])  # Does not change with azimuth, but stored for convenience
            TI_coherence[kk, jj] = np.median(Cxy2_trial[fmin_ind[jj]:fmax_ind[jj], jj])  # Here is where median infrasound-transverse coherence is calculated for each time/azimuth

    return ZI_coherence, TI_coherence


class SpectralEstimation:
    """ Spectral estimation class for tcm. """

    def __init__(self, data):
        """ Pre-allocate arrays and assignment of FFT-related variables. """
        # Sub-window size
        self.sub_window = int(np.round(data.winlensamp / 4))
        # Window overlap for spectral estimation [samples]
        self.noverlap = int(self.sub_window * 0.5)
        # Window for spectral estimation
        self.window = 'hann'
        # FFT frequency vector
        self.freq_vector = rfftfreq(self.sub_window, 1/data.sampling_rate)

        # Pre-allocate time vector
        self.t = np.full(data.nits, np.nan)

        # Pre-allocate cross spectral matrices (S)
        # Vertical and Infrasound
        self.S_zi = np.empty((len(self.freq_vector),
                              data.nits), dtype=complex)
        # Infrasound
        self.S_ii = np.full((len(self.freq_vector), data.nits), np.nan)
        # Vertical
        self.S_zz = np.full((len(self.freq_vector), data.nits), np.nan)
        # East-Infrasound
        self.S_ei = np.empty((len(self.freq_vector), data.nits), dtype=complex) # noqa
        # North-Infrasound
        self.S_ni = np.empty((len(self.freq_vector), data.nits), dtype=complex) # noqa
        # East-East
        self.S_ee = np.empty((len(self.freq_vector), data.nits), dtype=complex) # noqa
        # North-North
        self.S_nn = np.empty((len(self.freq_vector), data.nits), dtype=complex) # noqa
        # North-East
        self.S_ne = np.empty((len(self.freq_vector), data.nits), dtype=complex) # noqa
        # Magnitude Squared Coherence
        self.Cxy2 = np.full((len(self.freq_vector), data.nits), np.nan)

    def calculate_spectral_matrices(self, data):
        """ Calculate the cross spectral matrices. """

        """ Loop over time windows and calculate the cross spectrum
            at every frequency. """
        for jj in range(0, data.nits):
            # Get time from middle of window, except for the end.
            t0_ind = data.intervals[jj]
            tf_ind = data.intervals[jj] + data.winlensamp
            try:
                self.t[jj] = data.tvec[t0_ind + int(
                    np.round(data.winlensamp/2))]
            except:
                self.t[jj] = np.nanmax(self.t, axis=0)

            _, self.S_zi[:, jj] = csd(
                data.Z[t0_ind:tf_ind], data.Infra[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=self.sub_window, noverlap=self.noverlap)
            _, self.S_ii[:, jj] = np.real(csd(
                data.Infra[t0_ind:tf_ind], data.Infra[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=self.sub_window, noverlap=self.noverlap)) # noqa
            _, self.S_zz[:, jj] = np.real(csd(
                data.Z[t0_ind:tf_ind], data.Z[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=self.sub_window, noverlap=self.noverlap))
            _, self.S_ei[:, jj] = csd(
                data.E[t0_ind:tf_ind], data.Infra[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=self.sub_window, noverlap=self.noverlap)
            _, self.S_ni[:, jj] = csd(
                data.N[t0_ind:tf_ind], data.Infra[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=self.sub_window, noverlap=self.noverlap) # noqa
            _, self.S_ee[:, jj] = np.real(csd(
                data.E[t0_ind:tf_ind], data.E[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=self.sub_window, noverlap=self.noverlap))# noqa
            _, self.S_nn[:, jj] = np.real(csd(
                data.N[t0_ind:tf_ind], data.N[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=self.sub_window, noverlap=self.noverlap)) # noqa
            _, self.S_ne[:, jj] = csd(
                data.N[t0_ind:tf_ind], data.E[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=self.sub_window, noverlap=self.noverlap) # noqa
            _, self.Cxy2[:, jj] = coherence(
                data.Z[t0_ind:tf_ind], data.Infra[t0_ind:tf_ind],
                fs=data.sampling_rate, window=self.window,
                nperseg=self.sub_window, noverlap=self.noverlap)


class TCM:
    """ Perform transverse coherence minimization (TCM). """

    def __init__(self, data, spectrum):
        """ Pre-allocate arrays and assignment of TCM-related variables. """
        # Create azimuth vector [degrees]
        self.az_vector = np.array(np.arange(data.az_min, data.az_max + data.az_delta, data.az_delta)) # noqa

        # Number of samples in coherogram to smooth
        self.nsmth = 8

        # Get the closest frequency points to the preferred ones
        fmin_ind0 = np.argmin(np.abs(data.freq_min - spectrum.freq_vector))
        fmax_ind0 = np.argmin(np.abs(data.freq_max - spectrum.freq_vector))
        self.fmin_ind = np.array([fmin_ind0] * data.nits)
        self.fmax_ind = np.array([fmax_ind0] * data.nits)
        self.freq_min_array = np.array([data.freq_min] * data.nits)
        self.freq_max_array = np.array([data.freq_max] * data.nits)

        if data.search_2Hz:
            # Determine the number of 2 Hz bins in the frequency band:
            f_bandwidth = 2.0
            df = spectrum.freq_vector[1] - spectrum.freq_vector[0]
            f_bandwidth_increment = int(np.floor(f_bandwidth / df))
            n_bins = np.floor((data.freq_max - data.freq_min) / f_bandwidth)
            n_residual = ((data.freq_max - data.freq_min)
                          / f_bandwidth) - n_bins
            n_bins = int(n_bins)
            if n_bins > 1.0:
                for jj in range(0, data.nits):
                    f_min_iterable = fmin_ind0
                    f_max_iterable = fmin_ind0 + f_bandwidth_increment
                    coh_max = 0
                    for kk in range(0, n_bins):
                        coh_max_test = np.mean(spectrum.Cxy2[
                            f_min_iterable:f_max_iterable, jj])
                        if coh_max_test > coh_max:
                            coh_max = coh_max_test
                            self.fmin_ind[jj] = f_min_iterable
                            self.fmax_ind[jj] = f_max_iterable
                        else:
                            pass
                        f_min_iterable += f_bandwidth_increment
                        f_max_iterable += f_bandwidth_increment
                    if n_residual > 0.0:
                        f_min_iterable = fmax_ind0 - f_bandwidth_increment
                        f_max_iterable = fmax_ind0
                        coh_max_test = np.mean(spectrum.Cxy2[
                            f_min_iterable:f_max_iterable, jj])
                        if coh_max_test > coh_max:
                            coh_max = coh_max_test
                            self.fmin_ind[jj] = f_min_iterable
                            self.fmax_ind[jj] = f_max_iterable
            self.freq_min_array = spectrum.freq_vector[self.fmin_ind]
            self.freq_max_array = spectrum.freq_vector[self.fmax_ind]
        else:
            pass

        # Pre-allocate trial azimuth transverse-coherence matrices
        self.Cxy2_trial = np.empty((len(spectrum.freq_vector), data.nits))
        self.ZI_coherence = np.empty((len(self.az_vector), data.nits))
        self.TI_coherence = np.empty((len(self.az_vector), data.nits))

    def calculate_tcm_over_azimuths(self, data, spectrum):
        """ Calculate the  transverse coherence over all trial azimuths. """
        self.ZI_coherence, self.TI_coherence = _calculate_tcm_over_azimuths(data.nits, self.az_vector, spectrum.Cxy2, self.Cxy2_trial, spectrum.S_ni, spectrum.S_ei, spectrum.S_nn, spectrum.S_ne, spectrum.S_ee, spectrum.S_ii, self.ZI_coherence, self.TI_coherence, self.fmin_ind, self.fmax_ind) # noqa

    def find_minimum_tc(self, data, spectrum):
        """ Find the azimuths corresponding to the minimum transverse coherence. """
        # Apply some smoothing if desired
        self.bbv = np.full(data.nits - self.nsmth, 0, dtype='int')
        self.bbv2 = np.full(data.nits - self.nsmth, 0, dtype='int')
        self.median_coherence = np.full(data.nits - self.nsmth, np.nan)
        self.mean_phase_angle = np.full(data.nits - self.nsmth, 0, dtype='int')

        # Here are the 2 possible back-azimuths
        for jj in range(0, data.nits - self.nsmth):
            idx = np.argsort(np.sum(
                self.TI_coherence[:, jj:(jj + self.nsmth + 1)], 1))
            idx = np.sort(idx[0:2])
            self.bbv[jj] = idx[0]
            self.bbv2[jj] = idx[1]
            # Info on the amount of coherence
            self.median_coherence[jj] = np.max(np.median(
                self.ZI_coherence[:, jj:(jj + self.nsmth + 1)], axis=1))  # NEW

        # Resolve the 180 degree ambiguity by assuming retrograde motion
        self.Cxy2rz = np.empty((len(spectrum.freq_vector), data.nits), dtype=complex) # noqa
        self.Cxy2rz2 = np.empty((len(spectrum.freq_vector), data.nits), dtype=complex) # noqa
        self.Cxy2rza = np.empty((len(spectrum.freq_vector), data.nits)) # noqa
        self.Cxy2rza2 = np.empty((len(spectrum.freq_vector), data.nits)) # noqa
        for jj in range(0, data.nits - self.nsmth):
            t0_ind = data.intervals[jj]
            tf_ind = data.intervals[jj] + data.winlensamp
            _, self.Cxy2rz[:, jj] = csd(
                data.Z[t0_ind:tf_ind], data.N[t0_ind:tf_ind] * np.cos(
                    self.az_vector[self.bbv[jj]] * np.pi/180) + data.E[t0_ind:tf_ind] * np.sin(self.az_vector[self.bbv[jj]] * np.pi/180), fs=data.sampling_rate, window=spectrum.window, nperseg=spectrum.sub_window, noverlap=spectrum.noverlap) # noqa
            _, self.Cxy2rz2[:, jj] = csd(
                data.Z[t0_ind:tf_ind], data.N[t0_ind:tf_ind] * np.cos(
                    self.az_vector[self.bbv2[jj]] * np.pi/180) + data.E[t0_ind:tf_ind] * np.sin(
                    self.az_vector[self.bbv2[jj]] * np.pi/180), fs=data.sampling_rate, window=spectrum.window, nperseg=spectrum.sub_window, noverlap=spectrum.noverlap)
            self.Cxy2rza[:, jj] = np.angle(self.Cxy2rz[:, jj])
            self.Cxy2rza2[:, jj] = np.angle(self.Cxy2rz2[:, jj])
        # The time vector for the case of nonzero smoothing
        self.smvc = np.arange(((self.nsmth/2) + 1), (data.nits - (self.nsmth/2)) + 1, dtype='int') # noqa
        # The angle closest to -pi/2 is the azimuth, so the other
        # one is the back-azimuth
        tst1 = np.sum(self.Cxy2rza[self.fmin_ind[jj]:self.fmax_ind[jj], self.smvc] * spectrum.Cxy2[self.fmin_ind[jj]:self.fmax_ind[jj], self.smvc], axis=0)/np.sum(spectrum.Cxy2[self.fmin_ind[jj]:self.fmax_ind[jj], self.smvc], axis=0) # noqa
        tst2 = np.sum(self.Cxy2rza2[self.fmin_ind[jj]:self.fmax_ind[jj], self.smvc] * spectrum.Cxy2[self.fmin_ind[jj]:self.fmax_ind[jj], self.smvc], axis=0)/np.sum(spectrum.Cxy2[self.fmin_ind[jj]:self.fmax_ind[jj], self.smvc], axis=0) # noqa
        # Pick the angle the farthest from -pi/2
        self.baz_final = np.full(data.nits - self.nsmth, np.nan)
        for jj in range(0, len(self.bbv)):
            tst_ind = np.argmax(np.abs(np.array([tst1[jj], tst2[jj]]) - (-np.pi/2))) # noqa
            if tst_ind == 1:
                self.baz_final[jj] = self.az_vector[self.bbv[jj]]
            else:
                self.baz_final[jj] = self.az_vector[self.bbv2[jj]]

        # Convert azimuth to back-azimuth
        self.baz_final = (self.baz_final + 360) % 360

    def calculate_uncertainty(self, data, spectrum):
        """ Calculate uncertainty. """
        # See https://docs.obspy.org/_modules/obspy/signal/rotate.html
        Cxy2R = np.empty((len(spectrum.freq_vector), data.nits)) # noqa
        Cxy2T = np.empty((len(spectrum.freq_vector), data.nits)) # noqa
        for jj in range(0, data.nits - self.nsmth):
            t0_ind = data.intervals[jj]
            tf_ind = data.intervals[jj] + data.winlensamp
            R = -data.E[t0_ind:tf_ind] * np.sin(
                    self.baz_final[jj] * np.pi/180) - data.N[t0_ind:tf_ind] * np.cos(self.baz_final[jj] * np.pi/180) # noqa
            T = -data.E[t0_ind:tf_ind] * np.cos(
                    self.baz_final[jj] * np.pi/180) + data.N[t0_ind:tf_ind] * np.sin(self.baz_final[jj] * np.pi/180) # noqa
            _, Cxy2R[:, jj] = csd(R, R, fs=data.sampling_rate, scaling='spectrum', window=spectrum.window, nperseg=spectrum.sub_window, noverlap=spectrum.noverlap) # noqa
            _, Cxy2T[:, jj] = csd(T, T, fs=data.sampling_rate, scaling='spectrum', window=spectrum.window, nperseg=spectrum.sub_window, noverlap=spectrum.noverlap) # noqa

        # The time vector for the case of nonzero smoothing
        self.smvc = np.arange(((self.nsmth/2) + 1), (data.nits - (self.nsmth/2)) + 1, dtype='int') # noqa
        A2 = np.sum(Cxy2R[self.fmin_ind[jj]:self.fmax_ind[jj], self.smvc] * spectrum.Cxy2[self.fmin_ind[jj]:self.fmax_ind[jj], self.smvc], axis=0)/np.sum(spectrum.Cxy2[self.fmin_ind[jj]:self.fmax_ind[jj], self.smvc], axis=0) # noqa
        n2 = np.sum(Cxy2T[self.fmin_ind[jj]:self.fmax_ind[jj], self.smvc] * spectrum.Cxy2[self.fmin_ind[jj]:self.fmax_ind[jj], self.smvc], axis=0)/np.sum(spectrum.Cxy2[self.fmin_ind[jj]:self.fmax_ind[jj], self.smvc], axis=0) # noqa

        # Calculate sigma
        self.sigma = np.full_like(self.smvc, np.nan, dtype='float')
        idx_valid = np.where(A2 > 0.0)[0]
        self.sigma[idx_valid] = np.sqrt(
            (3 * n2[idx_valid]) / (16 * A2[idx_valid]))