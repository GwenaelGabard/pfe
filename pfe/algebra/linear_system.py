"""Linear system with sparse matrices"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


class Terms:
    """Sparse matrix formed by assembling a list of entries"""

    def __init__(self):
        """Default constructor"""
        self.i = []
        self.j = []
        self.v = []

    def clear(self):
        """Empty the list of entries in the matrix"""
        self.i = []
        self.j = []
        self.v = []

    def num(self):
        """Number of entries in the matrix

        Note that this is not necessarily the number of non-zero entries in the assembled matrix
        since several entries could be located at the same place in the matrix.

        :return: Number of entries
        :rtype: Positive integer
        """
        total = 0
        for ii in self.i:
            total += len(ii)
        return total

    def add(self, i, j, v):
        """Add more entries to the matrix

        :param i: List of row positions
        :type i: Sequence (typically a Numpy array)
        :param j: List of column positions
        :type j: Sequence (typically a Numpy array)
        :param v: List of values (potentially complex-valued)
        :type v: Sequence (typically a Numpy array)
        """
        self.i.append(i)
        self.j.append(j)
        self.v.append(v)

    def add_matrix(self, rows, cols, matrix):
        """Add a sub-matrix to the matrix

        :param rows: List of row positions
        :type rows: Sequence (typically a Numpy array)
        :param cols: List of column positions
        :type cols: Sequence (typically a Numpy array)
        :param matrix: Matrix (potentially complex-valued)
        :type matrix: Numpy array
        """
        self.i.append(np.repeat(rows, len(cols)))
        self.j.append(np.tile(cols, len(rows)))
        self.v.append(matrix.flatten())

    def matrix(self):
        """The assemble sparse matrix

        :return: The assembled sparse matrix
        :rtype: coo_matrix from scipy.sparse
        """
        ii = np.concatenate(self.i)
        jj = np.concatenate(self.j)
        vv = np.concatenate(self.v)
        return coo_matrix((vv, (ii, jj)))

    def assemble(self):
        """Assemble the entries in place"""
        M = self.matrix()
        M.sum_duplicates()
        self.i = [M.row.copy()]
        self.j = [M.col.copy()]
        self.v = [M.data.copy()]


class LinearSystem:
    r"""Linear system based on a sparse matrix assembled from element contributions.

    This class essentially represents a linear system of the form

    .. math::

       \mathbf{A}\mathbf{x} = \mathbf{b} \;,

    where :math:`\mathbf{A}` is a sparse matrix (potentially complex valued).
    The right-hand-side term :math:`\mathbf{b}` is a vector.

    TODO: Manage multiple right-hand-sides. Manage linear constraints on the system
    """

    def __init__(self):
        """Default constructor"""
        self.lhs = Terms()
        self.rhs = Terms()
        self.num_dofs = 0

    def clear(self):
        """Clear the right-hand-side and left-hand-side matrices as well as the number of
        non-zero entries
        """
        self.lhs.clear()
        self.rhs.clear()
        self.num_dofs = 0

    def get_dofs(self, num):
        """Declare additional degrees of freedom in the linear system

        :param num: Number of DOFs to add
        :type num: Positive integer
        :return: A sequence of DOF numbers
        :rtype: A Numpy array of positive integers
        """
        total = np.sum(num)
        dofs = self.num_dofs + np.arange(total)
        self.num_dofs += total
        return dofs

    def assemble(self):
        """Assemble the left- and right-hand sides of the system"""
        self.lhs.assemble()
        self.rhs.assemble()

    def solve(self):
        """Solve the sparse linear system using the scipy.sparse.linalg method

        :return: The solution vector
        :rtype: Numpy array (potentially complex valued)
        """
        A = self.lhs.matrix().tocsc()
        A.resize((self.num_dofs, self.num_dofs))
        b = self.rhs.matrix()
        b.resize((self.num_dofs, 1))
        b = b.todense()
        x = spsolve(A, b)
        return x
