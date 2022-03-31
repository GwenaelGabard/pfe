r"""
Interpolation
=============

The ``interpolation`` module contains classes implementing different interpolation methods on a
given mesh.
The following interpolations are available:

Quadratic Lagrange interpolation
--------------------------------

.. autoclass:: pfe.interpolation.Lagrange2
   :members:

"""

from .lagrange2 import Lagrange2

__all__ = ["Lagrange2"]
