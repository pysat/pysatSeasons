.. _install:

Installation
============

The following instructions will allow you to install pysatSeasons.


.. _install-prereq:

Prerequisites
-------------

.. image:: images/logo.png
    :width: 150px
    :align: right
    :alt: pysatSeasons Logo, Calendar Icon with pysat logo and SEASONS at top.


pysatSeasons uses common Python modules, as well as modules developed by and for
the Space Physics community.  This module officially supports Python 3.6+.

 ============== =================
 Common modules Community modules
 ============== =================
  matplotlib    pysat
  numpy
  pandas
  xarray
 ============== =================


.. _install-opt:

Installation Options
--------------------

1. Clone the git repository
::


   git clone https://github.com/pysat/pysatSeasons.git


2. Install pysatSeasons:
   Change directories into the repository folder and run the setup.py file.
   There are a few ways you can do this:

   A. Install on the system (root privileges required)::


        sudo python3 setup.py install
   B. Install at the user level::


        python3 setup.py install --user
   C. Install with the intent to develop locally::


        python3 setup.py develop --user
