import numpy as np
from scipy.signal import csd
from scipy.fft import rfftfreq


def calc_csm(dat, fs, FMIN, FMAX):
    r""" Calculate a cross-spectral matrix from a data array

        Args:
            dat: An mxn data array with n traces of m points.
            fs: data sampling rate
            FMIN: frequency lower bound
            FMAX: frequency upper bound

        Returns:
            csm: A cross spectral matrix
            freq: Corresponding frequencies
    """

    # Set/Calculate FFT and S parameters
    nperseg, n = np.shape(dat)
    # Calculate trace pairs
    # idx = [(i, j) for i in range(0, n) for j in range(i+1, n)]
    idx = [(i, j) for i in range(n) for j in range(n)]
    nfft = np.power(2, int(np.ceil(np.log2(nperseg))))  # pad fft
    # Calculate fft frequencies
    freq = rfftfreq(nfft, 1/fs)
    finds = np.where((freq >= FMIN) & (freq <= FMAX))
    freq = freq[np.where((freq >= FMIN) & (freq <= FMAX))]

    # Calculate S
    S = np.full((n, n, len(freq)), np.nan, dtype=complex)
    for jj in range(0, len(idx)):
        y1 = dat[:, idx[jj][0]]
        y2 = dat[:, idx[jj][1]]
        [f, Pxy] = csd(y1, y2, fs, window='hann', nfft=nfft)
        Pxy = Pxy[finds]
        S[idx[jj][0], idx[jj][1], :] = Pxy

    return S, freq


def calc_freq_num(winlen, fs, fmin, fmax):
    r""" Calculates the number of frequencies for pre-allocation

        Args:
            WINLEN:
            fs:
            FMIN:
            FMAX:

        Returns:
            num_freq: The number of frequencies
    """
    m = int(winlen*fs)  # samples
    # nperseg = m * 2 - 1 # Update sample # for CC trace length
    nperseg = m
    nfft = np.power(2, int(np.ceil(np.log2(nperseg))))  # pad fft
    freq = rfftfreq(nfft, 1/fs)
    freq = freq[np.where((freq >= fmin) & (freq <= fmax))]

    return len(freq)


def calc_gp(S, freq):
    r""" Calculate the magnitude squared coherence and phase spectrum

        Args:
            S: Cross spectral matrix. S is assumed to be 2 x 2 for now.
            freq: Frequencies of interest

        Returns:
            gam2: magnitude squared coherence
            phas2: phase spectrum

    """
    sww = S[0, 0, :]
    spp = S[1, 1, :]
    swp = S[0, 1, :]

    # Calculate the magnitude squared coherence
    gam2 = np.real(np.divide(np.multiply(swp, np.conj(swp)),
                             np.multiply(sww, spp)))

    # Calculate the Cospectrum
    cwp = np.real(swp)
    # Calculate the Quadrature Spectrum
    qwp = np.imag(swp)
    # Calculate the Phase Spectrum (degrees)
    phas2 = np.arctan2(-qwp, cwp) * (180/np.pi)
    # Convert phase spectrum from (-180,180) to (0, 360)
    phas2 = (phas2 + 360) % 360

    return gam2, phas2
