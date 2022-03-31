r"""Functions providing the quadrature points on the reference elements"""
import os
import numpy as np

line_table = None
triangle_table = None


def line(order):
    r"""Provide the points :math:`u_n` and weights :math:`w_n`
    (with :math:`n=0, ..., N-1`) for Gauss quadrature on the reference line
    defined by:

    .. math::

       0<u<1 .

    When using :math:`N` integration points, the quadrature rule will be exact
    up to order :math:`K=2N-1`.

    *Usage:*

    .. code-block:: python

        from pfe.quadrature import line
        u, weights = line(order)

    *Input:*

    * ``order``: The maximum order :math:`K` of integration.

    *Output:*

    * ``u``: A numpy array with the coordinates of the integration points.
    * ``weights``: A numpy array with the weights of the integration points.
    """
    global line_table
    if line_table is None:
        line_table = np.load(
            os.path.join(os.path.dirname(__file__), "line_quadrature.npy"),
            allow_pickle=True,
        )
    u, weight = line_table[order]
    return (u.copy(), weight.copy())


def triangle(order):
    r"""Provide the coordinates :math:`(u_n, v_n)` and weights :math:`w_n`
    (with :math:`n=0, ..., N-1`) for Gauss quadrature on the reference triangle
    defined by:

    .. math::

       u>0, v>0, u+v<1 .

    When using :math:`N` integration points, the quadrature rule will be exact
    up to order :math:`K=2N-1`.

    *Usage:*

    .. code-block:: python

        from pfe.quadrature import triangle
        uv, weights = triangle(order)

    *Input:*

    * ``order``: The maximum order :math:`K` of integration.

    *Output:*

    * ``uv``: A numpy array with the coordinates of the integration points (each row is a
        coordinate, each column is a point).
    * ``weights``: A numpy array with the weights of the integration points.
    """
    global triangle_table
    if triangle_table is None:
        triangle_table = np.load(
            os.path.join(os.path.dirname(__file__), "triangle_quadrature.npy"),
            allow_pickle=True,
        )
    uv, weight = triangle_table[order]
    return (uv.copy(), weight.copy())
