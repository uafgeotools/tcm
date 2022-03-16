import matplotlib.pyplot as plt
from matplotlib import dates


def tcm_plot(data, CSM):
    fig, axs = plt.subplots(3, 1, sharex='col')
    # Vertical component of seismic trace
    axs[0].plot(data.tvec, data.Z, c='k')
    axs[0].set_ylabel('Displacement \n [m]')
    # Magnitude squared coherence
    sc0 = axs[1].pcolormesh(CSM.t, CSM.freq_vector, CSM.Cxy2,
                            cmap='plasma', shading='auto')
    axs[1].axis('tight')
    axs[1].set_xlim(CSM.t[0], CSM.t[-1])
    axs[1].set_ylim(CSM.freq_vector[0], CSM.freq_vector[-1])
    axs[1].set_ylabel('Frequency \n [Hz]', fontsize=12)
    p1 = axs[1].get_position()
    cbaxes1 = fig.add_axes([0.92, p1.y0, 0.02, p1.height])
    hc1 = plt.colorbar(sc0, orientation="vertical",
                       cax=cbaxes1, ax=axs[1])
    hc1.set_label('Max Weighted \n Coherence')
    sc0.set_clim(0.0, 1.0)

    # Back-azimuth Estimate
    sc1 = axs[2].scatter(CSM.t[CSM.smvc], CSM.baz_final, c=CSM.aa2, cmap='plasma')
    # axs[2].set_ylim(-180, 180)
    axs[2].axhline(-52, c='k', linestyle=':')
    axs[2].set_ylim(-180, 180)
    axs[2].set_ylabel('Back-Azimuth \n [Deg.]', fontsize=12)
    p1 = axs[2].get_position()
    cbaxes1 = fig.add_axes([0.92, p1.y0, 0.02, p1.height])
    hc1 = plt.colorbar(sc1, orientation="vertical",
                       cax=cbaxes1, ax=axs[2])
    hc1.set_label('Max Weighted \n Coherence')
    sc1.set_clim(0.0, 1.0)

    axs[2].xaxis_date()
    axs[2].tick_params(axis='x', labelbottom='on')
    axs[2].fmt_xdata = dates.DateFormatter('%HH:%MM')
    axs[2].xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
    axs[2].set_xlabel('UTC Time')
    # fig.savefig('Python_TCM_Example.png', bbox_inches='tight', dpi=300, facecolor="w") # noqa

    return fig, axs
