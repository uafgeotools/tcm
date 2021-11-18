from setuptools import setup
import os

# https://github.com/readthedocs/readthedocs.org/issues/5512#issuecomment-475073310
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = ['waveform_collection']

setup(
    name='tcm',
    url='https://github.com/jwbishop/tcm',
    install_requires=INSTALL_REQUIRES
)
