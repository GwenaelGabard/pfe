"""Quadratic triangle element in 2D"""
import numpy as np
from pfe.shape import T6
from pfe.quadrature import triangle
from .locate import locateT6


class Triangle6_2D:
    """Quadratic triangle element in 2D"""

    def __init__(self, nodes):
        """Default constructor

        :param nodes: Physical coordinates of the nodes forming the element
        :type nodes: A Numpy array of coordinates
        """
        self.nodes = nodes.T
        self.shape = T6

    def position(self, uv):
        """The physical coordinates of a point on the element

        :param uv: Coordinates on the reference element
        :type uv: A Numpy array
        :return: Physical coordinates of the point
        :rtype: A Numpy array
        """
        xy = self.shape.S(uv) @ self.nodes
        return xy.T

    def metric(self, uv):
        """Compute the metric between reference and physical coordinates

        :param uv: Reference coordinates of the point to consider
        :type uv: A Numpy array
        :return: Jacobian and its determinant of the mapping
        :rtype: A tuple containing two Numpy arrays and a scalar
        """
        dSdu, dSdv = self.shape.dS(uv)
        dxydu = dSdu @ self.nodes
        dxydv = dSdv @ self.nodes
        detJ = dxydu[:, 0] * dxydv[:, 1] - dxydu[:, 1] * dxydv[:, 0]
        return (dxydu, dxydv, detJ)

    def metric_from_order(self, order):
        """Compute the metric between reference and physical coordinates

        :param order: Quadrature order defining the points to consider
        :type order: A positive integer
        :return: Jacobian and its determinant of the mapping
        :rtype: A tuple containing two Numpy arrays and a scalar
        """
        _, dSdu, dSdv = self.shape.from_order(order)
        dxydu = dSdu @ self.nodes
        dxydv = dSdv @ self.nodes
        detJ = dxydu[:, 0] * dxydv[:, 1] - dxydu[:, 1] * dxydv[:, 0]
        return (dxydu, dxydv, detJ)

    def du_to_dx(self, dfdu, dfdv, dxydu, dxydv, det):
        """Convert gradients from reference to physical coordinate systems

        :param dfdu: Gradients w.r.t. to the reference coordinate u
        :type dfdu: A Numpy array
        :param dfdv: Gradients w.r.t. to the reference coordinate v
        :type dfdv: A Numpy array
        :param dxydu: Metric between reference and physical coordinates
        :type dxydu: A Numpy array
        :param dxydv: Metric between reference and physical coordinates
        :type dxydv: A Numpy array
        :param det: Determinant of the Jacobian matrix
        :type det: A Numpy array
        :return: Gradients w.r.t. to the physical coordinates
        :rtype: A tuple with two Numpy arrays
        """
        dfdx = (dxydv[:, 1] * dfdu.T - dxydu[:, 1] * dfdv.T) / det
        dfdy = (-dxydv[:, 0] * dfdu.T + dxydu[:, 0] * dfdv.T) / det
        return (dfdx.T, dfdy.T)

    def gradient(self, uv, dfdu, dfdv):
        """Convert gradients from reference to physical coordinate systems

        :param uv: Reference coordinates for the points to consider
        :type uv: A Numpy array
        :param dfdu: Gradients w.r.t. to the reference coordinate u
        :type dfdu: A Numpy array
        :param dfdv: Gradients w.r.t. to the reference coordinate v
        :type dfdv: A Numpy array
        :return: Gradients w.r.t. to the physical coordinates
        :rtype: A tuple with two Numpy arrays
        """
        dxydu, dxydv, detJ = self.metric(uv)
        return self.du_to_dx(dfdu, dfdv, dxydu, dxydv, detJ)

    def ref_integration(self, order):
        """Quadrature scheme on the reference element

        :param order: Quadrature order
        :type order: A positive integer
        :return: Positions and weights of the quadrature scheme
        :rtype: A tuple with two Numpy arrays
        """
        return triangle(order)

    def integration(self, order):
        """Quadrature scheme on the physical element

        :param order: Quadrature order
        :type order: A positive integer
        :return: Positions and weights of the quadrature scheme
        :rtype: A tuple with two Numpy arrays
        """
        uv, weight = triangle(order)
        _, _, detJ = self.metric(uv)
        weight *= detJ
        return (uv, weight)

    def locate_point(self, p):
        """Compute the reference coodinates of a physical point

        :param p: Coordinates of the point to locate
        :type p: A Numpy array
        :return: The reference coordinates of the point if inside the element. None otherwise.
        :rtype: A tuple with two scalars, or None
        """
        u, v = locateT6(self.nodes[:, 0].copy(), self.nodes[:, 1].copy(), p[0], p[1])
        if u >= 0 and v >= 0 and (u + v) <= 1:
            return np.array([[u], [v]])
        return None
