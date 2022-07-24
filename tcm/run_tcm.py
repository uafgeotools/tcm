from tcm.classes import tcm_classes, tcm_data_class


def run_tcm(st, freq_min, freq_max, window_length,
            window_overlap, az_min=0.0, az_max=359.0, az_delta=1.0):
    """ Process stream data with the transverse coherence minimization algorithm (TCM).

    Args:
        st: Obspy stream object. Assumes response has been removed.
        freq_min (float): Minimum frequency for analysis.
        freq_max (float): Maximum frequency for analysis.
        window_length (float): Window length in seconds.
        window_overlap (float): Window overlap in the range (0.0 - 1.0).
        az_min (float): Minimum (lower bound) azimuth for search in degrees.
        az_max (float): Maximum (upper bound) azimuth for search in degrees.
        az_delta (float): Azimuth increment for search in degrees.

    Returns:
        (tuple)
            A tuple of transverse coherence minimization parameters:
            ``baz`` (array): back-azimuth estimate (degrees)
            ``sigma`` (array): estimated back-azimuth uncertainty (degrees)
            ``smoothed_time`` (array): time vector for ``baz`` and ``sigma`` that is smaller due to smoothing.
            ``freq_vector`` (array):
            ``time_vector`` (array):
            ``Cxy2`` (array):

    """
    # Create object to hold data and pre-process
    data = tcm_data_class.DataBin(freq_min, freq_max,
                                  window_length, window_overlap,
                                  az_min, az_max, az_delta)
    data.build_data_arrays(st)

    # Create cross-spectral matrix object
    CSM = tcm_classes.Spectral(data)
    # Calculate spectra, cross-spectra, and vertical component coherence
    CSM.calculate_spectral_matrices(data)
    # Calculate the transverse coherence over all trial azimuths
    CSM.calculate_tcm_over_azimuths(data)
    # Find the coherence minima and apply the retrograde assumption
    baz, sigma = CSM.find_minimum_tc(data)

    return CSM.baz, CSM.sigma, CSM.t[CSM.smvc], CSM.freq_vector, CSM.t, CSM.Cxy2  # noqa