r"""Quadratic shape functions for the line."""
import numpy as np


class L3:
    r"""
    Quadratic shape functions for the line.

    These three functions are defined as follows with the variable :math:`u`
    on the reference line element :math:`0<u<1`:

    .. math::

       \phi_0(u)=1+u(2u-3), \phi_1(u)=u(2u-1), \phi_2(u)=4u(1-u) \;.

    These are nodal shape functions associated to the nodes:

    .. math::

       u_0=0, u_1=1, u_2=0.5 .

    Note that this is a purely static class.
    """

    order = 2

    @staticmethod
    def S(u):
        r"""Values of the shape functions.

        :param u: Vector of coordinates :math:`u` on the element (expecting :math:`0<u<1`).
        :return: A numpy array with 3 columns and as many rows as elements in `u`.
        """
        u = np.array(u, copy=False, ndmin=1)
        f = np.zeros((u.shape[1], 3))
        f[:, 0] = u * (2.0 * u - 3.0) + 1.0
        f[:, 1] = u * (2.0 * u - 1.0)
        f[:, 2] = u * (1.0 - u) * 4.0
        return f

    @staticmethod
    def dS(u):
        r"""First derivatives of the shape functions.

        :param u: Vector of coordinates :math:`u` on the element (expecting :math:`0<u<1`).
        :return: A numpy array with 3 columns and as many rows as elements in `u`.
        """
        u = np.array(u, copy=False, ndmin=1)
        f = np.zeros((u.shape[1], 3))
        f[:, 0] = 4.0 * u - 3.0
        f[:, 1] = 4.0 * u - 1.0
        f[:, 2] = 4.0 - 8.0 * u
        return f
