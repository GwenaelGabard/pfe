.. pfe documentation master file, created by
   sphinx-quickstart on Tue Feb 16 22:43:25 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Python Finite Elements
======================

The codebase is organised around several sub-modules:

* :doc:`quadrature`: Integration rules on reference elements.
* :doc:`shape`: Basic shape functions on reference elements.
* :doc:`algebra`: Discrete models, typically the linear system of equations.
* :doc:`mesh`: The finite element mesh.
* :doc:`geometry`: The shape of the different elements (linear triangle, quadratic line, etc).
* :doc:`interpolation`: The interpolation of fields (either for the solutions and the parameters).
* :doc:`model`: The finite element model.

A number of physical models are provided:

* :doc:`lpe_2d`
* :doc:`helmholtz_2d`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
