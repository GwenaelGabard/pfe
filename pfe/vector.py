"""Storage for a vector fo degrees of freedom"""
import numpy as np


class Vector:
    """Storage vor a vector of degrees of freedom

    Unlike a field which inherently depends on position, a Vector only contains a presrcibed
    number of unknowns which can be declared as degrees of freedom in a finite-element model.
    This can be used for instance for the amplitudes of acoustic modes in a duct connected to the
    finite-element domain.
    """

    def __init__(self, data):
        """Constructor

        :param data: An int to specify the number of entries in the vector, or a sequence
            containing the values of the entries in the vector
        :type data: A positive integer or a sequence
        """
        if isinstance(data, int):
            self.length = data
            self.known = False
        else:
            self.data = np.asarray(data)
            self.length = len(self.values)
            self.known = True
        self.dofs = None

    def declare_dofs(self, system):
        """Declare the entries of the vector as degrees of freedom in a system

        :param system: The system to which we declare the degrees of freedom
        :type system: A pfe linear system
        """
        self.dofs = system.get_dofs(self.length)

    def num_dofs(self):
        """The number of degrees of freedom

        This is the same as the length of the vector.

        :return: The number of degrees of freedom
        :rtype: A positive integer
        """
        return len(self.dofs)

    def values(self):
        """The content of the vector

        :return: The content of the vector if known, otherwise returns None
        :rtype: A Numpy array or None
        """
        if self.known:
            return self.data
        return None
