User Notes
================
Here we list a few notes for seismoacoustic processing with the transverse coherence minimization (TCM) method.


Spectral Search
------------------
From our experience, the method works best if applied in an approximately 2 Hz frequency band. The option `search_2Hz` is a boolean parameter that defaults to `False`. If set to `True`, then the back-azimuth will be estimated only using data from the 2 Hz frequency band with the largest magnitude-squared coherence (:math:`{C_{xy}^2}`). This frequency band is determined at every time step, and the width of the band can be changed (e.g. 2 Hz :math:`\rightarrow` 3 Hz) by modifying the `f_bandwidth` variable in the `TCM` class.


Uncertainty Quantification
---------------------------------------
This method estimates a back-azimuth value, which wraps around 360 :math:`{^\circ}`. This means, for example, that 1 :math:`{^\circ}` and 359 :math:`{^\circ}` are only separated by 2 :math:`{^\circ}`. For this kind of directional data, a Von Mises distribution can be used to characterize the underlying distribution (Mardia & Jupp,  2009).

For the Von Mises distribution, the variance in the distribution (:math:`{\sigma^2}`) is a function of the parameter :math:`{\kappa}` as

.. math:: \sigma^2 = 1 - \frac{I_1(\kappa)}{I_0(\kappa)},

where :math:`{I_1}` and  :math:`{I_0}` are modified Bessel functions of the first and zeroth order, respectively. For large :math:`{\kappa}`, :math:`{\sigma^2 \approx \frac{3}{8\kappa}}`.
We use Taylor series to match the curvature of the magnitude squared coherence of the infrasound microphone and the radial seismic component and the Von Mises distribution. Thus,

.. math:: \kappa \approx 2\frac{A_r^2}{n^2},

where :math:`{A_r}` is the amplitude of the radial seismic component and :math:`{n}` is the noise amplitude.

Using the estimated back-azimuth, the code automatically calculates an uncertainty estimate by first rotating the horizontal seismic components to radial and transverse directions. The transverse component is used as an estimate of the noise amplitude, and we return an uncertainty as

.. math:: \sigma \approx \sqrt{\frac{3 n^2}{16 A_r^2}}.

If we view :math:`{\frac{A_r^2}{n^2}}` as a signal-to-noise ratio squared, then we see that our uncertainty decreases as the signal-to-noise ratio increases.

