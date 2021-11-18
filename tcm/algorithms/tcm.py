import numpy as np

def tcm(st, winlen, winover, fmin, fmax):
    r""" Returns the angle that minizes the coherence between and infrasound trace and the transverse component of a seismometer

    Args:
        st: Obspy stream file
        winlen: processing window length [sec]
        winover: processing window overlap (0, 1)
        fmin: Minimum processing frequency [Hz]
        fmax: Maximum processing frequency [Hz]

    Returns:
        t: time
        baz: back-azimuth to tcm source
    """

    # Pull processing parameters from the stream file.
    tvec = st[0].times('matplotlib')
    nchans = len(st)
    npts = st[0].stats.npts
    fs = st[0].stats.sampling_rate

    # check that all traces have the same length
    if len(set([len(tr) for tr in st])) != 1:
        raise ValueError('Traces in stream must have same length!')

    # Convert window length to samples
    winlensamp = int(winlen*fs)
    sampinc = int((1-winover)*winlensamp)
    its = np.arange(0, npts, sampinc)
    # We need all the data windows to be the same length
    # The last data block is discarded.
    nits = len(its) - int(np.ceil(winlensamp/sampinc))

    # Pre-allocate data arrays.
    t = np.full(nits, np.nan)
    baz = np.full((nits,), np.nan)

    # Loop through the time series and rotate the components
    az_test = np.linspace(0, 360, 720)
    drot = 0.5
    counter = 0

    for jj in range(nits):
        # Get time from middle of window, except for the end.
        ptr = int(its[jj]), int(its[jj] + winlensamp)
        try:
            t[jj] = tvec[ptr[0]+int(winlensamp/2)]
        except:
            t[jj] = np.nanmax(t, axis=0)

        test_baz = np.fill(len(az_test), np.nan)

        for kk in range(0, len(az_test)):
            st.rotate(drot)
            # Store data traces in an array for processing.
            data = np.empty((npts, nchans))
            for ii in range(0, nchans):
                data[:, ii] = st[ii].data

            # Calculate the cross-spectral matrix S
            S, freq = pa.calc_csm(data[ptr[0]:ptr[1], :], fs, fmin, fmax)

            # Calculate the coherence and phase spectrum
            gtest, _ = pa.calc_gp(S, freq)
            test_baz[kk] = np.mean(gtest)

        # Print progress
        baz[jj] = np.argmin(test_baz)*drot
        counter += 1
        print('{:.1f}%'.format((counter / nits) * 100), end='\r')

    print('\nDone\n')

    return t, baz
