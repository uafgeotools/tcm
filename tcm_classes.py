
class DataBin:
    """ Data container for TCM processing"""

    def __init__(self, freq_min, freq_max, window_length, az_min,
                 az_max, az_delta, assume_retrograde=True):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.window_length = window_length
        self.az_min = az_min
        self.az_max = az_max
        self.az_delta = az_delta
        self.assume_retrograde = assume_retrograde

    def build_data_arrays(self, st):
        # Assumes all traces have the same sample rate
        self.sampling_rate = st[0].stats.sampling_rate
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


class CSM:
    """ A Cross Spectral Matrix class"""

    def __init__(self):
        pass
