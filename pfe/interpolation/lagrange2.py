"""Quadratic Lagrange interpolation on a finite-element mesh"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pfe.shape import L3, T6


from_type = {9: T6, 8: L3}

sub_t6 = np.array([[0, 3, 5], [3, 4, 5], [3, 1, 4], [4, 2, 5]], dtype="uint32")
sub_from_type = {9: sub_t6}


class Lagrange2:
    r"""Quadratic Lagrange interpolation

    The class ``Lagrange2`` represents a field interpolated on a mesh using a second-order
    Lagrange interpolation.
    Depending on the nature of each individual element in the mesh (triangle, tetrahedron, etc)
    this class will use the corresponding quadratic shape functions to interpolate the field
    within the element.

    ``Lagrange2`` can be used for an unknown field, such as the pressure field governed by the
    Helmholtz equation.
    In this case, an instance can be created with only the mesh it will be interpolated on:

    .. code-block:: python

        from pfe.interpolation import Lagrange2
        field = Lagrange2(mesh)

    This field should then be added to the dictionary of unknowns for the model:

    .. code-block:: python

        model.fields["phi"] = field

    In this example, this field is named ``phi``.
    The unkown values of the field will then be declared as degrees of freedom of the discrete
    model.
    Once the model is solved and a solution is obtained, values of the field can be computed using
    this solution.

    ``Lagrange2`` can also be used for a known field, such as a parameter appearing in the
    differential equation to be solved, or for a source term.
    In this case, to create an instance we have to provide the mesh used for the interpolation and
    a function that returns the value of the field at a given point.
    In the example below, we define a quadratic interpolation of the function :math:`x+2y` on a
    given mesh:

    .. code-block:: python

        from pfe.interpolation import Lagrange2

        def value(xy):
            return(xy[0,:]+2*xy[1,:])

        field = Lagrange2(mesh, value)

    This field can then be used as a parameter in a model. To this end it must be declared in the
    dictionary of parameters of the model:

    .. code-block:: python

        model.parameters["c0"] = field

    In this example, the parameter name is ``c0``.
    """

    def __init__(self, domain, value=None):
        """Default constructor

        :param domain: The finite-element mesh
        :type domain: A pfe.Mesh instance
        :param value: The value of the field interpolated on the mesh, defaults to None
        :type value: A function taking a Numpy array of coordinates as input, optional
        """
        self.domain = domain
        self.known = False
        self.dofs = None
        self.order = 2
        if value is not None:
            self.store_values(value)
            self.known = True

    def store_values(self, value):
        """Compute and store the values of the interpolated field at the nodes of the mesh

        :param value: The value of the field interpolated on the mesh
        :type value: A function taking a Numpy array of coordinates as input
        """
        self.values = value(self.domain.nodes())
        self.dofs = np.arange(len(self.values))

    def declare_dofs(self, system):
        """Declare the unknown values of the interpolated field as degrees of freedom of a system

        :param system: The system to which we declare the degrees of freedom
        :type system: A pfe linear system
        """
        self.dofs = system.get_dofs(self.domain.num_nodes())

    def basis(self, element):
        """The interpolation basis within a given element of the mesh

        :param element: The tag of the requested element
        :type element: A positive integer
        :return: An interpolation basis
        :rtype: An object from pfe.shape
        """
        etype = self.domain.element_type(element)
        return from_type[etype]

    def num_dofs(self):
        """The number of degrees of freedom needed to define this field

        :return: The number of degrees of freedom for this field
        :rtype: A positive integer
        """
        return len(self.dofs)

    def element_dofs(self, tag):
        """The numbers of the degrees of freedom contributing to a given element

        :param tag: The tag of the requested element
        :type tag: A positive integer
        :return: A sequence of DOF numbers
        :rtype: A Numpy array of positive integers
        """
        return self.dofs[self.domain.element_node_tags(tag)]

    def get_value(self, tag, uvw=None, xyz=None):
        """Compute the value of the field at a given point

        :param tag: The tag of the element involved
        :type tag: A positive integer
        :param uvw: The coordinates of the point on the reference element, defaults to None
        :type uvw: A Numpy array, optional
        :param xyz: The physical coordinates of the point, defaults to None
        :type xyz: A numpy array, optional
        :return: The value of the field
        :rtype: A scalar (possibly complex valued)
        """
        if uvw is not None:
            basis = self.basis(tag)
            dofs = self.element_dofs(tag)
            values = basis.S(uvw) @ self.values[dofs]
            return values
        return None

    def plot(self, solution=None):
        """Generate a plot of the field on the mesh

        :param solution: The values of the degrees of freedom
        :type solution: A numpy array
        :return: The elements added to the plot
        :rtype: Matplotlib triangulation
        """
        if solution is None:
            solution = self.values
        elements = self.domain.get_elements(2)
        T = []
        for e in elements:
            sub = sub_from_type[self.domain.element_type(e)]
            nodes = self.domain.element_node_tags(e)
            T.append(nodes[sub])
        T = np.concatenate(T)
        if self.domain.num_dim == 3:
            x, y, _ = self.domain.coordinates
        elif self.domain.num_dim == 2:
            x, y = self.domain.coordinates
        tri = matplotlib.tri.Triangulation(x, y, T)
        values = solution[self.dofs]
        plt.tricontourf(tri, values)
        return tri

    def sample(self, probes, solution):
        """Compute the values of the field at several points

        :param probes: The list of probes on the mesh
        :type probes: An instance of pfe.mesh.Probes (generated by pfe.Mesh.locate_points)
        :param solution: The values of the degrees of freedom
        :type solution: A numpy array
        :return: A sequence of values of the field
        :rtype: A Numpy array
        """
        num_probes = len(probes.elements)
        f = np.zeros((num_probes,), dtype=solution.dtype)
        f.fill(np.nan)
        for n in range(num_probes):
            tag = probes.elements[n]
            if tag >= 0:
                basis = self.basis(tag)
                dofs = self.element_dofs(tag)
                f[n] = basis.S(probes.positions[n]) @ solution[dofs]
        return f

    def sample_grad(self, probes, solution):
        """Compute the gradient of the field at several points

        :param probes: The list of probes on the mesh
        :type probes: An instance of pfe.mesh.Probes (generated by pfe.Mesh.locate_points)
        :param solution: The values of the degrees of freedom
        :type solution: A numpy array
        :return: A sequence of values of the field gradients
        :rtype: A Numpy array
        """
        num_probes = len(probes.elements)
        dfdx = np.zeros((num_probes,), dtype=solution.dtype)
        dfdx.fill(np.nan)
        dfdy = dfdx.copy()
        for n in range(num_probes):
            tag = probes.elements[n]
            if tag >= 0:
                basis = self.basis(tag)
                geometry = self.domain.element_geometry(tag)
                dxydu, dxydv, detJ = geometry.metric(probes.positions[n])
                dSdu, dSdv = basis.dS(probes.positions[n])
                dSdx, dSdy = geometry.du_to_dx(dSdu, dSdv, dxydu, dxydv, detJ)
                dofs = solution[self.element_dofs(tag)]
                dfdx[n] = dSdx @ dofs
                dfdy[n] = dSdy @ dofs
        return dfdx, dfdy
