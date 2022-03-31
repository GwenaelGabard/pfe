r"""Quadratic shape functions for the triangle."""
import os
import numpy as np


LOOKUP_T6 = None


class T6:
    r"""
    Quadratic shape functions for the triangle.

    These six functions are defined with the variables :math:`(u,v)`
    on the reference triangle element defined as

    .. math::
        u>0, v>0, u+v<1 \;.

    These shape functions are defined as follows:

    .. math::
        \phi_0(u,v)=u(2u+4v-3)+v(2v-3)+1 \;,

    .. math::
        \phi_1(u,v)=u(2u-1) \;,

    .. math::
        \phi_2(u,v)=v(2v-1) \;,

    .. math::
        \phi_3(u,v)=4u(1-u-v) \;,

    .. math::
        \phi_4(u,v)=4uv \;,

    .. math::
        \phi_5(u,v)=-4uv+4v(1-v) \;.

    These are nodal shape functions associated to the nodes:

    .. math::

        (u,v)_0=(0,0),
        (u,v)_1=(1,0),
        (u,v)_2=(0,1),
        (u,v)_3=(0.5,0),
        (u,v)_4=(0.5,0.5),
        (u,v)_5=(0,0.5).

    Note that this is a purely static class.
    """

    order = 2

    @staticmethod
    def S(uv):
        r"""Values of the shape functions.

        :param uv: Array of coordinates on the element (the first row is :math:`u`,
            the second row is :math:`v`).
        :return: A numpy array with 6 columns and as many rows as elements in `uv`.
        """
        u = uv[0, :]
        v = uv[1, :]
        f = np.zeros((len(u), 6))
        f[:, 0] = u * (2.0 * u + 4.0 * v - 3.0) + v * (2.0 * v - 3.0) + 1.0
        f[:, 1] = u * (2.0 * u - 1.0)
        f[:, 2] = v * (2.0 * v - 1.0)
        f[:, 3] = u * (-4.0 * u - 4.0 * v + 4.0)
        f[:, 4] = 4.0 * u * v
        f[:, 5] = -4.0 * u * v + 4.0 * v * (1.0 - v)
        return f

    @staticmethod
    def dS(uv):
        r"""First derivatives of the shape functions.

        :param uv: Array of coordinates on the element (the first row is :math:`u`,
            the second row is :math:`v`).
        :return: Two numpy arrays with 6 columns and as many rows as elements in `uv`,
            corresponding to the derivative along the first and second coordinate, respectively.
        """
        u = uv[0, :]
        v = uv[1, :]
        du = np.zeros((len(u), 6))
        du[:, 0] = 4.0 * u + 4.0 * v - 3.0
        du[:, 1] = 4.0 * u - 1.0
        du[:, 2] = 0.0
        du[:, 3] = -8.0 * u - 4.0 * v + 4.0
        du[:, 4] = 4.0 * v
        du[:, 5] = -4.0 * v
        dv = np.zeros((len(u), 6))
        dv[:, 0] = 4.0 * u + 4.0 * v - 3.0
        dv[:, 1] = 0.0
        dv[:, 2] = 4.0 * v - 1.0
        dv[:, 3] = -4.0 * u
        dv[:, 4] = 4.0 * u
        dv[:, 5] = -4.0 * u - 8.0 * v + 4.0
        return (du, dv)

    @staticmethod
    def from_order(order):
        r"""Values of shape functions based on look-up tables.

        These tables are generated and stored in a ``.npy`` file prior to the module installation
        using the script ``make_basis.py``.

        :param order: The polynomial order of integration. From this information we have the
            number and position of the integration points.
        :return: Three numpy arrays with 6 columns and as many rows as integration points,
            corresponding to the values and derivative along the first and second coordinate,
            respectively.
        """
        global LOOKUP_T6
        if LOOKUP_T6 is None:
            LOOKUP_T6 = np.load(
                os.path.join(os.path.dirname(__file__), "table_T6.npy"),
                allow_pickle=True,
            )
        S, dSdu, dSdv = LOOKUP_T6[order]
        return (S.copy(), dSdu.copy(), dSdv.copy())
