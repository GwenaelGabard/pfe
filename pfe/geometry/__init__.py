r"""
Geometry
========

The ``geometry`` module gathers the geometries of the different finite elements (lines, triangles,
tetrahedra, etc).
The following elements are available:

Two dimensions
--------------

Quadratic line (3 nodes)
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pfe.geometry.Line3_2D
   :members:

Quadratic triangle (6 nodes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pfe.geometry.Triangle6_2D
   :members:
"""

from .line3_2d import Line3_2D
from .triangle6_2d import Triangle6_2D

__all__ = ["Line3_2D", "Triangle6_2D"]
