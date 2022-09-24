"""Point"""


class Point:
    """Point"""

    def __init__(self, node):
        """Default constructor

        :param node: Physical coordinates of the point
        :type node: A Numpy array of coordinates
        """
        self.node = node

    def position(self):
        """The physical coordinates of the point

        :return: Physical coordinates of the point
        :rtype: A Numpy array
        """
        return self.node

    def locate_point(self, _):
        """Locate a point inside the element

        No point can be contained inside a point.
        This is just a stand-in function for other types of elements.
        """
        return None
