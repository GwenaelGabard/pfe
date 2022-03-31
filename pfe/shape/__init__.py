r"""
Shape
=====

The ``shape`` module gathers the definitions of the shape functions on reference elements.

Linear shape functions
-------------------------

TBC

Quadratic shape functions
-------------------------

Line
~~~~

.. autoclass:: pfe.shape.L3
   :members:

Triangle
~~~~~~~~

.. autoclass:: pfe.shape.T6
   :members:
"""


from .l3 import L3
from .t6 import T6

__all__ = ["L3", "T6"]
