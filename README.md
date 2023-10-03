tcm
=============
This package consists of tools for the transverse coherence minimization (TCM) seismo-acoustic analysis method. TCM can be used to estimate the back-azimuth of infrasound signals that are recorded on an infrasound sensor and a colocated three-component seismometer. When using this code we ask you to cite the following paper, which also provides details on the method and some examples:

Bishop, J. W., Haney, M. M., Fee, D., Matoza, R. S., Mckee, K. F., & Lyons, J. J. (2023). Back-Azimuth Estimation of Air-to-Ground Coupled Infrasound from Transverse Coherence Minimization, 249–258. https://doi.org/10.1785/0320230023.

Installation
---------------

*Here are install instructions for an example conda environment.*

We recommend using conda and creating a new conda environment such as
```
conda create -n tcm_env python=3 obspy numba
```
or from the provided `.yml` file as
```
conda env create -f environment.yml
```

Information on conda environments (and more) is available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

After setting up the conda environment, [install](https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs) the package by running the terminal commands:
```
conda activate tcm_env
git clone https://github.com/uafgeotools/tcm
cd tcm
pip install -e .
```
The final command installs the package in "editable" mode, which means that you
can update it with a simple `git pull` in your local repository. This install
command only needs to be run once.


Dependencies
--------------------
*For example:*

Python packages:
* [Python3](https://docs.python.org/3/)
* [ObsPy](http://docs.obspy.org/)
* [Numba](http://numba.pydata.org)


Example
-----------
See the included *example.py*.


Usage
---------
Import the package like any other python package, ensuring the correct environment is active. For example,
```
$ conda activate tcm_env
$ python
>>> import tcm
```
See  the attached documentation for more information.

Authors
-------

(_Alphabetical order by last name._)

Jordan W. Bishop <br>
David Fee <br>
Matthew M. Haney <br>


Acknowledgements and Distribution Statement
-------------------------------------------

This work was supported by the Defense Threat Reduction Agency Nuclear Arms Control Technology program under contract HQ003421F0112. Approved for public release; distribution is unlimited.
