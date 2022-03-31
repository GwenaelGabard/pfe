r"""
Algebra
=======

The ``algebra`` module contains the representation of discrete models, typically the linear system
obtained once a finite element model is discretised.

.. autoclass:: pfe.algebra.LinearSystem
   :members:

"""

from .linear_system import LinearSystem

__all__ = ["LinearSystem"]
