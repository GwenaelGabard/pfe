"""Linear system with sparse matrices"""
import numpy as np
from scipy.sparse import coo_matrix, bmat
from scipy.sparse.linalg import spsolve
from numpy.linalg import qr, solve


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


class Constraint:
    """A linear constraint to impose on the linear system
    """
    def __init__(self, dof, coef, rhs):
        """Default constructor

        :param dof: list of dof numbers
        :type dof: list or numpy array
        :param coef: list of coefficients
        :type coef: list or numpy array
        :param rhs: rhs value of the constraint, defaults to 0.0
        :type rhs: float or complex, optional
        """
        self.dof = dof
        self.coef = coef
        self.rhs = rhs


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
        self.constraints = []

    def clear(self):
        """Clear the right-hand-side and left-hand-side matrices as well as the number of
        non-zero entries
        """
        self.lhs.clear()
        self.rhs.clear()
        self.num_dofs = 0
        self.constraints = []

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

    def add_constraint(self, dof, coef, rhs=0.0):
        """Add a linear constraint

        :param dof: list of dof numbers
        :type dof: list or numpy array
        :param coef: list of coefficients
        :type coef: list or numpy array
        :param rhs: rhs value of the constraint, defaults to 0.0
        :type rhs: float or complex, optional
        """
        self.constraints.append(Constraint(dof,coef,rhs))

    def assemble(self):
        """Assemble the left- and right-hand sides of the system"""
        self.lhs.assemble()
        self.rhs.assemble()

    def solve(self):
        """Solve the sparse linear system using the scipy.sparse.linalg method

        :return: The solution vector
        :rtype: Numpy array (potentially complex valued)
        """
        # Check if constraints are defined
        if self.constraints:
            return self.csolve()
        # Solve without constraint
        A = self.lhs.matrix().tocsc()
        A.resize((self.num_dofs, self.num_dofs))
        b = self.rhs.matrix()
        b.resize((self.num_dofs, 1))
        b = b.todense()
        x = spsolve(A, b)
        return x

    def csolve(self):
        """Solve the sparse linear system together with linear constraints

        :return: The solution vector
        :rtype: Numpy array (potentially complex valued)
        """
        # Number of constraints
        m = len(self.constraints)
        # Create a list of dof involved in the constraints
        nc = []
        for c in self.constraints:
            nc += list(c.dof)
        nc = np.unique(nc).astype(int)
        p = len(nc)
        # Create a list of dof not involved in the constraints
        nf = np.arange(self.num_dofs)
        nf[nc] = -1
        nf = np.unique(nf)[1:]
        # Create the matrix of constraints
        t = Terms()
        d = []
        for n, c in enumerate(self.constraints):
            t.add([n]*len(c.dof), c.dof, c.coef)
            d.append(c.rhs)
        C = t.matrix().tocsr()[:,nc].toarray()
        d = np.array(d)
        # QR factorization of C
        Q, R = qr(C.T, 'complete')
        Q2 = Q[:,m:]
        # Solve for the constrained subspace
        x1 = Q[:,:m]@solve(R[:m,:].T, d)
        # Assemble the lhs and rhs of the system
        A = self.lhs.matrix().tocsr()
        A.resize((self.num_dofs, self.num_dofs))
        b = self.rhs.matrix()
        b.resize((self.num_dofs, 1))
        b = b.toarray().flatten()
        # Build the reduced linear system (without the constrained subspace)
        Afc = A[np.ix_(nf, nc)]
        Acc = A[np.ix_(nc, nc)]
        A2 = bmat([[Q2.T@Acc@Q2, Q2.T@A[np.ix_(nc, nf)]], [Afc@Q2, A[np.ix_(nf, nf)]]]).tocsc()
        b2 = Q2.T@(b[nc]-Acc@x1)
        bf = b[nf] - Afc@x1
        bc = np.vstack((b2[:,None],bf[:,None]))
        # Solve the reduced system
        x = spsolve(A2, bc)
        # Build the full solution vector
        xx = np.zeros((self.num_dofs,), dtype=complex)
        xx[nc] = x1 + Q2@x[:len(b2)]
        xx[nf] = x[len(b2):]
        return xx
