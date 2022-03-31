r"""Script to generate and write the look-up table for the shape functions"""
import os
import sys
import numpy as np
from t6 import T6

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from quadrature import triangle


def make_T6(max_order, filename):
    r"""Compute and store the values of the shape functions for different quadrature orders"""
    table = []
    for order in range(max_order):
        uv, _ = triangle(order)
        S = T6.S(uv)
        dSdu, dSdv = T6.dS(uv)
        table.append((S, dSdu, dSdv))
    data = np.array(table, dtype=object)
    np.save(filename, data, allow_pickle=True)


if __name__ == "__main__":
    make_T6(40, "table_T6.npy")
