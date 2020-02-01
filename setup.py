#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019, RS
# Full license can be found in License.md
# -----------------------------------------------------------------------------

import os
import codecs
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, 'description.txt'), encoding='utf-8') as f:
    long_description = f.read()
version_filename = os.path.join('pysatSeasons', 'version.txt')
with codecs.open(os.path.join(here, version_filename)) as version_file:
    version = version_file.read().strip()

# change setup.py for readthedocs - commented for now
# on_rtd = os.environ.get('READTHEDOCS') == 'True'
install_requires = ['pysat', 'numpy', 'pandas<0.25', 'matplotlib']


# Run setup

setup(name='pysatSeasons',
      version=version,
      url='http://github.com/pysat/pysatSeasons',
      author='Russell Stoneback, Jeff Klenzing',
      author_email='rstoneba@utdallas.edu',
      description='Seasonal analysis routines for pysat',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      classifiers=[
          "Development Status :: 4 - Beta",
          "Topic :: Scientific/Engineering :: Astronomy",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Scientific/Engineering :: Atmospheric Science",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Natural Language :: English",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
      ],
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      )
