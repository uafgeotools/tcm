tcm
=============
This package consists of tools for the transverse coherence minimization (tcm) seismo-acoustic analysis method.

Installation
---------------

*Here are install instructions for an example conda environment. For consistency, we encourage all interfacing packages in uafgeotools to use conda environments.*

We recommend using conda and creating a new conda environment such as:
```
conda create -n uafinfra python=3 obspy
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
* [ObsPy](http://docs.obspy.org/)


Example
-----------
See the included *example script here.py*.


Usage
---------
Import the package like any other python package, ensuring the correct environment
is active. For example,
```
$ conda activate uafinfra
$ python
>>> import tcm
```
*Mention documentation here. Perhaps point to the example.py file*

Authors
-------

(_Alphabetical order by last name._)

Jordan W. Bishop <br>
David Fee <br>
Matthew M. Haney <br>