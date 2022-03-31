"""Imposed normal velocity boundary condition for the LPE in 2D"""


class Velocity:
    r"""Implements the boundary condition for an imposed normal velocity for the
    Linearised Potential Equation in 2D:

    .. math::

       \frac{\partial\phi}{\partial n}=V \;,

    where :math:`V` is the normal velocity imposed on the boundary :math:`\Gamma`.
    Note that this is defined using the outgoing normal to the computational domain.
    The mean flow is assumed to be tangential to the boundary:
    :math:`\mathbf{u}_0\cdot\mathbf{n}=0`.
    To implement this boundary condition, the following term is added to the right-hand side
    of the linear system:

    .. math::

       - \int_\Gamma \rho_0 \overline{\psi}V \,\mathrm{d}\Gamma \;,

    where :math:`\psi` is the test function.

    In the following code snippet we apply this boundary condition with a constant normal velocity
    on the boundary defined as the mesh group 4:

    .. code-block:: python

        from pfe.models.lpe_2d import Velocity

        model.add_term(Velocity(mesh.group(4)), Constant(1.0))

    A complete example is provided in the Jupyter notebook ``examples/lpe_2d/velocity``.
    """

    def __init__(self, domain, velocity):
        """Constructor

        :param domain: The finite-element model
        :type domain: An instance of pfe.Model
        :param velocity: The normal velocity of the surface
        :type velocity: An instance of pfe.Constant, pfe.Function or an interpolated field,
            optional
        """
        self.domain = domain
        self.velocity = velocity

    def assemble(self, model, system):
        """Assemble the element matrices

        :param model: The finite-element model
        :type model: An instance of pfe.Model
        :param system: The algebraic system ton contribute to
        :type system: An instance of a class from pfe.algebra
        """
        elements = self.domain.get_elements(dim=1)
        for element in elements:
            Fe = self.terms(model, element)
            dof = model.fields["phi"].element_dofs(element)
            system.rhs.add(dof, dof * 0, Fe.flatten())

    def terms(self, model, e):
        """Compute the element matrix for a single element

        :param model: The finite-element model
        :type model: An instance of pfe.Model
        :param e: The element tag
        :type e: A positive integer
        :return: The element matrix
        :rtype: A Numpy array
        """
        basis = model.fields["phi"].basis(e)
        geometry = self.domain.element_geometry(e)
        u, weights = geometry.integration(2 * basis.order)
        xy = geometry.position(u)
        phi = basis.S(u)
        rho0 = model.parameters["rho0"].get_value(e, u, xy)
        V = self.velocity.get_value(e, u, xy)
        Fe = -phi.T @ ((weights * rho0) * V)
        return Fe
