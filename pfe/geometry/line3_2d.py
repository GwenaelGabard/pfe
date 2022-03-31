"""Quadratic line element in 2D"""
import numpy as np
from pfe.shape import L3
from pfe.quadrature import line


class Line3_2D:
    """Quadratic line element in 2D"""

    def __init__(self, nodes):
        """Default constructor

        :param nodes: Physical coordinates of the nodes forming the element
        :type nodes: A Numpy array of coordinates
        """
        self.nodes = nodes.T
        self.shape = L3

    def position(self, u):
        """The physical coordinates of a point on the element

        :param u: Coordinate on the reference element
        :type u: A scalar between 0 and 1
        :return: Physical coordinates of the point
        :rtype: A Numpy array
        """
        xy = self.shape.S(u) @ self.nodes
        return xy.T

    def metric(self, u):
        """Compute the metric between reference and physical coordinates

        :param u: Reference coordinate of the point to consider
        :type u: A scalar between 0 and 1
        :return: Jacobian and its determinant of the mapping
        :rtype: A tuple containing a Numpy array and a scalar
        """
        J = self.shape.dS(u) @ self.nodes
        detJ = np.sqrt(J[:, 0] ** 2 + J[:, 1] ** 2)
        return (J, detJ)

    def tangent(self, u):
        """The unit tangent vector along the element at a given point

        :param u: Reference coordinate of the point to consider
        :type u: A scalar between 0 and 1
        :return: The unit tangent vector
        :rtype: A Numpy array
        """
        J, detJ = self.metric(u)
        return np.divide(J, detJ[:, None])

    def normal(self, u):
        """The unit normal vector to the element at a given poin

        :param u: Reference coordinate of the point to consider
        :type u: A scalar between 0 and 1
        :return: The unit normal vector
        :rtype: A Numpy array
        """
        t = self.tangent(u)
        n = np.zeros_like(t)
        n[:, 0] = t[:, 1]
        n[:, 1] = -t[:, 0]
        return n

    def integration(self, order):
        """Quadrature scheme on the element for a given integration order

        :param order: The quadrature order
        :type order: A positive integer
        :return: The positions and weights of the integration points
        :rtype: A tuple containing two Numpy arrays
        """
        u, weight = line(order)
        _, detJ = self.metric(u)
        weight *= detJ
        return (u, weight)

    def basis_from_u(self, basis, u):
        """Interpolation functions and their derivative in physical coordinates

        :param basis: The basis to compute
        :type basis: A class from pfe.shape
        :param u: The reference coordinates of the points to consider
        :type u: A Numpy array of values between 0 and 1
        :return: The values and the derivatives of the basis functions
        :rtype: A tuple with two Numpy arrays
        """
        _, detJ = self.metric(u)
        S = basis.S(u)
        dSdu = basis.dS(u)
        dSdtau = dSdu / detJ[:, None]
        return (S, dSdtau)

    def basis_from_order(self, basis, order):
        """Interpolation functions and their derivative in physical coordinates

        :param basis: The basis to compute
        :type basis: A class from pfe.shape
        :param order: The quadrature order defining the points to consider
        :type order: A positive integer
        :return: The values and the derivatives of the basis functions
        :rtype: A tuple with two Numpy arrays
        """
        u, _ = line(order)
        _, detJ = self.metric(u)
        S = basis.S(u)
        dSdu = basis.dS(u)
        dSdtau = dSdu / detJ[:, None]
        return (S, dSdtau)

    def locate_point(self, _):
        """Locate a point inside the element

        No point can be contained inside this line element.
        This is just a stand-in function for other types of elements.
        """
        return None
