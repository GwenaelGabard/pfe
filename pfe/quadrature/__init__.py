r"""
Quadrature
==========

The ``quadrature`` module provides Gauss quadrature rules on reference elements (line, triangle,
etc).
For each reference element, a function is defined that provides the positions and weights of the
Gauss integration points.
These functions take as single argument the maximum polynomial order of the integration rule.

By construction, the integration points are always located inside the reference elements and their
weights are strictly positive.

A general guideline is that the reference elements are always defined between 0 and 1.


Line
----

.. automodule:: pfe.quadrature.main.line
   :members:

Triangle
--------

.. automodule:: pfe.quadrature.main.triangle
   :members:

Implementation note
-------------------

To improve performance, the sets of quadrature rules are not computed on the fly but taken from
look-up tables.
These tables are generated and stored in ``.npy`` files prior to the module installation using the
script ``make_quadrature.py``.
"""

from .main import line, triangle

__all__ = ["line", "triangle"]
