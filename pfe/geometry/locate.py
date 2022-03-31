"""Functions to locate a point inside an element"""
import numpy as np
from numba import jit


@jit()
def locateT3(x, y, px, py):
    """Compute the reference coordinates of a point w.r.t to a T3 element (linear triangle)

    :param x: x coordinates of the triangle vertices
    :type x: A Numpy array
    :param y: y coordinates of the triangle vertices
    :type y: A Numpy array
    :param px: x coordinate of the point to locate
    :type px: A scalar
    :param py: y coordinate of the point to locate
    :type py: A scalar
    :return: The reference coordinates of the point
    :rtype: A tuple with two scalars
    """
    a0 = x[1] - x[0]
    a1 = y[1] - y[0]
    b0 = x[2] - x[0]
    b1 = y[2] - y[0]
    d0 = px - x[0]
    d1 = py - y[0]
    u = (d0 * b1 - d1 * b0) / (a0 * b1 - a1 * b0)
    v = (d0 * a1 - d1 * a0) / (b0 * a1 - b1 * a0)
    return (u, v)


@jit()
def locateT6(x, y, px, py):
    """Compute the reference coordinates of a point w.r.t to a T6 element (quadratic triangle)

    This method uses an iterative scheme since this is a non-linear problem.

    :param x: x coordinates of the triangle vertices
    :type x: A Numpy array
    :param y: y coordinates of the triangle vertices
    :type y: A Numpy array
    :param px: x coordinate of the point to locate
    :type px: A scalar
    :param py: y coordinate of the point to locate
    :type py: A scalar
    :return: The reference coordinates of the point
    :rtype: A tuple with two scalars
    """
    epsilon = 0.7
    tolerance = 1.0e-10
    max_iter = 100
    u, v = locateT3(x[:3], y[:3], px, py)
    for _ in range(max_iter):
        N = np.array(
            [
                u * (2.0 * u + 4.0 * v - 3.0) + v * (2.0 * v - 3.0) + 1.0,
                u * (2.0 * u - 1.0),
                v * (2.0 * v - 1.0),
                u * (-4.0 * u - 4.0 * v + 4.0),
                4.0 * u * v,
                -4.0 * u * v + 4.0 * v * (1.0 - v),
            ]
        )
        dNdu = np.array(
            [
                4.0 * u + 4.0 * v - 3.0,
                4.0 * u - 1.0,
                0.0,
                -8.0 * u - 4.0 * v + 4.0,
                4.0 * v,
                -4.0 * v,
            ]
        )
        dNdv = np.array(
            [
                4.0 * u + 4.0 * v - 3.0,
                0.0,
                4.0 * v - 1.0,
                -4.0 * u,
                4.0 * u,
                -4.0 * u - 8.0 * v + 4.0,
            ]
        )
        dx = px - N.dot(x)
        dy = py - N.dot(y)
        du = epsilon * (np.dot(dNdu, x) * dx + np.dot(dNdu, y) * dy)
        dv = epsilon * (np.dot(dNdv, x) * dx + np.dot(dNdv, y) * dy)
        u += du
        v += dv
        if np.sqrt(du * du + dv * dv) < tolerance:
            break
    return (u, v)
