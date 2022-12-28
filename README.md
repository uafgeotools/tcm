tcm
=============
This package consists of tools for the transverse coherence minimization (tcm) seismo-acoustic analysis method.

Installation
---------------

*Here are install instructions for an example conda environment.*

We recommend using conda and creating a new conda environment such as:
```
conda create -n uafinfra python=3 obspy numba
```

If you have a previously created `uafinfra` environment, you may need to install the `numba <http://numba.pydata.org>`__  package with

```
conda install --name uafinfra numba
```

Information on conda environments (and more) is available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

After setting up the conda environment, [install](https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs) the package by running the terminal commands:
```
conda activate uafinfra
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

_uafgeotools_ packages:
* [_waveform_collection_](https://github.com/uafgeotools/waveform_collection)

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
$ conda activate uafinfra
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