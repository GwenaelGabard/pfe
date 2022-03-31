"""Script to generate and store the look-up table for the Gauss quadrature schemes"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def line_quad(order):
    """Gauss quadrature on a line

    :param order: Required quadrature order
    :type order: Positive integer
    :return: Positions and weights of the quadrature points
    :rtype: Tuple with two Numpy arrays
    """
    u, weight = leggauss(int((order + 1) / 2.0) + 1)
    u = (u + 1.0) / 2.0
    weight *= 0.5
    return (u, weight)


def make_line(max_order, filename):
    """Compute and store all Gauss quadrature schemes on a line up to a given order

    :param max_order: Maximum quadrature order to compute
    :type max_order: Positive integer
    :param filename: The file name to store the quadrature schemes
    :type filename: String
    """
    table = []
    for order in range(max_order):
        u, weight = line_quad(order)
        u = u[None, :]
        table.append((u, weight))
    np.save(filename, table, allow_pickle=True)


def triangle_quad(order):
    """Gauss quadrature on a triangle

    :param order: Required quadrature order
    :type order: Positive integer
    :return: Positions and weights of the quadrature points
    :rtype: Tuple with two Numpy arrays
    """
    N = int((order + 1) / 2.0) + 1
    uref, wref = leggauss(N)
    uref = (uref + 1.0) / 2.0
    wref *= 0.5
    u = []
    v = []
    weight = []
    for n in range(N):
        u = np.append(u, np.full((N,), uref[n]))
        v = np.append(v, uref * (1.0 - uref[n]))
        weight = np.append(weight, np.full((N,), wref[n] * wref * (1.0 - uref[n])))
    uv = np.vstack((u, v))
    return (uv, weight)


def make_triangle(max_order, filename):
    """Compute and store all Gauss quadrature schemes on a triangle up to a given order

    :param max_order: Maximum quadrature order to compute
    :type max_order: Positive integer
    :param filename: The file name to store the quadrature schemes
    :type filename: String
    """
    table = []
    for order in range(max_order):
        uv, weight = triangle_quad(order)
        table.append((uv, weight))
    np.save(filename, table, allow_pickle=True)


if __name__ == "__main__":
    make_line(40, "line_quadrature.npy")
    make_triangle(40, "triangle_quadrature.npy")
