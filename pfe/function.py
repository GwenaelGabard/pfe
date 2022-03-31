"""Store a field given by a function of position"""


class Function:
    """Store a field given by a function of position"""

    def __init__(self, function, order=3):
        """Constructor

        :param function: The function computing the known field
        :type function: A Python function taking as argument a Numpy array containing the physical
            coordinates of points where to evaluate the field. It should return a Numpy array.
        :param order: The order of quadrature to use for this field, defaults to 3
        :type order: Positive integer, optional
        """
        self.function = function
        self.order = order

    def get_value(self, e, _, xyz):
        """Compute the value of the field at a given point

        :param e: Unused (needed for consistency with other classes in pfe)
        :param xyz: Physical coordinates of the points where the field will be evaluated
        :type xyz: A Numpy array
        :return: The values of the function
        :rtype: A Numpy array
        """
        return self.function(xyz)
