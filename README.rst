.. role:: raw-html-m2r(raw)
   :format: html

Transverse Coherence Minimization (TCM)
========================================

This package contains the Python3 version of Matt Haney's Matlab TCM code.

Installation
------------

*Here are install instructions for an example conda environment. For consistency, we encourage all interfacing packages in uafgeotools to use conda environments.*

We recommend using conda and creating a new conda environment such as:

.. code-block::

   conda create -n uafinfra python=3 obspy

Information on conda environments (and more) is available `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

After setting up the conda environment, `install <https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs>`_ the package by running the terminal commands:

.. code-block::

   conda activate uafinfra
   git clone https://github.com/uafgeotools/polarization_analysis
   cd polarization_analysis
   pip install -e .

The final command installs the package in "editable" mode, which means that you
can update it with a simple ``git pull`` in your local repository. This install
command only needs to be run once.

Dependencies
------------

*For example:*

*uafgeotools* packages:


* `_waveform\ *collection* <https://github.com/uafgeotools/waveform_collection>`_

Python packages:


* `ObsPy <http://docs.obspy.org/>`_

Example
-------

See the included *example script here.py*.

Usage
-----

Import the package like any other python package, ensuring the correct environment
is active. For example,

.. code-block::

   $ conda activate uafinfra
   $ python
   >>> import tcm_py

*Mention documentation here. Perhaps point to the example.py file*