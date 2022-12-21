# Class type hinting only available in Python 3.9+
from __future__ import annotations

from obspy.core.stream import Stream
from tcm.classes import tcm_classes, tcm_data_class


def run_tcm(st: type[Stream], freq_min: float, freq_max: float, window_length: float, window_overlap: float, az_min: float = 0.0, az_max: float = 359.0, az_delta: float = 1.0) -> tuple: # noqa
    """ Process stream data with the transverse coherence minimization algorithm (TCM).

    Args:
        st: An obspy stream object. Assumes any preprocessing has already occured.
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
            ``baz`` (array): back-azimuth estimate (degrees).
            ``sigma`` (array): estimated back-azimuth uncertainty (degrees).
            ``smoothed_time`` (array): time vector for ``baz`` and ``sigma`` that accounts for smoothing.
            ``freq_vector`` (array): frequency vector for Cxy2.
            ``time_vector`` (array): time vector for Cxy2.
            ``Cxy2`` (array): Magnitude-squared coherence between the vertical displacement (Z) and the infrasound pressure.
            ``mean_coherence`` (array):  Mean coherence value across smoothed back-azimuth estimate.

    """
    # Create object to hold data and pre-process
    data = tcm_data_class.DataBin(freq_min, freq_max,
                                  window_length, window_overlap,
                                  az_min, az_max, az_delta)
    data.build_data_arrays(st)

    # Create cross-spectral matrix object
    CSM = tcm_classes.SpectralEstimation(data)
    # Calculate spectra, cross-spectra, and vertical component coherence
    CSM.calculate_spectral_matrices(data)
    # Create the TCM object
    TCM = tcm_classes.TCM(data, CSM)
    # Calculate the transverse coherence over all trial azimuths
    TCM.calculate_tcm_over_azimuths(data, CSM)
    # Find the coherence minima and apply the retrograde assumption
    TCM.find_minimum_tc(data, CSM)
    # Estimate uncertainty
    TCM.calculate_uncertainty(data, CSM)

    return TCM.baz_final, TCM.sigma, CSM.t[TCM.smvc], CSM.freq_vector, CSM.t, CSM.Cxy2, TCM.mean_coherence  # noqa