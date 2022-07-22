import matplotlib.pyplot as plt
from matplotlib import dates

cm = 'magma_r'
c_lim = [0.4, 1.0]

def tcm_plot(data, CSM):
    fig, axs = plt.subplots(3, 1, sharex='col', figsize=(8, 11))
    # Vertical component of seismic trace
    axs[0].plot(data.tvec, data.Z, c='k')

    axs[0].set_ylabel('Displacement \n [m]')
    # Magnitude squared coherence
    sc0 = axs[1].pcolormesh(CSM.t, CSM.freq_vector, CSM.Cxy2,
                            cmap=cm, shading='auto')
    axs[1].axis('tight')
    axs[1].set_xlim(CSM.t[0], CSM.t[-1])
    axs[1].set_ylim(data.freq_min, data.freq_max)
    axs[1].set_ylabel('Frequency \n [Hz]', fontsize=12)
    p1 = axs[1].get_position()
    sc0.set_clim(c_lim)

    # Back-azimuth Estimate
    sc1 = axs[2].scatter(CSM.t[CSM.smvc], CSM.baz_final, c=CSM.aa2, cmap=cm,
                         edgecolors='k', lw=0.3)
    axs[2].scatter(CSM.t[CSM.smvc], CSM.baz_final + CSM.sigma,
                   c='gray', marker='_', linestyle=':')
    axs[2].scatter(CSM.t[CSM.smvc], CSM.baz_final - CSM.sigma,
                   c='gray', marker='_', linestyle=':')
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
