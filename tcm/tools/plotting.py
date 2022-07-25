import matplotlib.pyplot as plt
from matplotlib import dates


def tcm_plot(st, freq_min, freq_max, baz, time_smooth, freq_vector, time, Cxy2, mean_coherence):  # noqa
    """ Return a plot of the TCM results.

    Plots (a) the vertical seismic trace, (b) the magnitude squared coherence
     between the infrasound signal and the vertical seismic displacement,
     and (c) the estimated back-azimuth from the TCM algorithm.

    Args:
        st (stream): Obspy stream.
        freq_min (float):
        freq_max (float):
        baz (array):
        time_smooth (array):
        freq_vector (array):
        time (array):
        Cxy2 (array):
        mean_coherence (array):

    Returns:
        (tuple):
            ``fig``: Output figure handle.
            ``axs``: Output axis handle.
    """
    # Specify the colormap.
    cm = 'magma_r'
    # Colorbar/y-axis limits for the vertical coherence
    c_lim = [0.4, 1.0]
    # Specify the time vector for plotting the trace.
    tvec = st[0].times('matplotlib')

    fig, axs = plt.subplots(3, 1, sharex='col', figsize=(8, 11))
    # Vertical component of seismic trace (displacement)
    axs[0].plot(tvec, st[0], c='k')

    axs[0].set_ylabel('Displacement \n [m]')
    # Magnitude squared coherence
    sc0 = axs[1].pcolormesh(time, freq_vector, Cxy2,
                            cmap=cm, shading='auto')
    axs[1].axis('tight')
    axs[1].set_xlim(time[0], time[-1])
    axs[1].set_ylim(freq_min, freq_max)
    axs[1].set_ylabel('Frequency \n [Hz]', fontsize=12)
    p1 = axs[1].get_position()
    sc0.set_clim(c_lim)

    # Back-azimuth Estimate
    sc1 = axs[2].scatter(time_smooth, baz, c=mean_coherence, cmap=cm,
                         edgecolors='k', lw=0.3)
    axs[2].set_ylim(0, 360)
    axs[2].set_ylabel('Back-Azimuth \n [Deg.]', fontsize=12)
    p2 = axs[2].get_position()
    sc1.set_clim(c_lim)

    cbot = p2.y0
    ctop = p1.y1
    cbaxes = fig.add_axes([0.92, cbot, 0.02, ctop-cbot])
    hc = plt.colorbar(sc0, cax=cbaxes)
    hc.set_label('Max Weighted Coherence')

    axs[2].xaxis_date()
    axs[2].tick_params(axis='x', labelbottom='on')
    axs[2].fmt_xdata = dates.DateFormatter('%HH:%MM')
    axs[2].xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
    axs[2].set_xlabel('UTC Time')

    return fig, axs
