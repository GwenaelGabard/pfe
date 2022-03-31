"""Storage for a constant value"""
import numpy as np


class Constant:
    """Storage for a constant value"""

    def __init__(self, value):
        """Constructor

        :param value: The value of the constant
        :type value: A scalar
        """
        self.value = value
        self.order = 1

    def get_value(self, _=None, uvw=None, xyz=None):
        """The value of the constant

        Note that only the shapes of uvw and xyz are used here, their content is not used.

        :param e: The element tag where the value is requested, defaults to None
        :type e: A positive integer, optional
        :param uvw: The reference coordinates where the constant is requested, defaults to None
        :type uvw: A Numpy array, optional
        :param xyz: The physical coordinates where the constant is requested, defaults to None
        :type xyz: A Numpy array, optional
        :return: The value of the constant
        :rtype: A scalar or a Numpy array (depending on the content of uvw or xyz)
        """
        if xyz is not None:
            return np.full((xyz.shape[1],), self.value)
        if uvw is not None:
            return np.full((uvw.shape[1],), self.value)
        return self.value
