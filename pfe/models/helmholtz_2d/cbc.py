"""Characteristics boundary condition for the Helmholtz equation in 2D"""
import numpy as np
from pfe.constant import Constant


class CBC:
    r"""Implements a characteristic boundary condition for the Linearised Potential
    Equation in 2D.
    This condition allows to specify the incoming characteristic of the solution which in the
    following way:

    .. math::

       \frac{\partial\phi}{\partial n} + \mathrm{i}k^+\phi = g \;,

    where :math:`n` is the outgoing normal on the boundary :math:`\Gamma`.
    The wavenumber :math:`k^+=\omega/(c_0+\mathbf{u}_0\cdot\mathbf{n})` is that of the
    outgoing plane wave normal to the boundary.
    The source term :math:`g` specifies the incoming characteristics.

    To implement this boundary condition, the following term is added to the left-hand side of the
    linear system:

    .. math::

       -\int_\Gamma \mathrm{i}\omega \frac{\rho_0}{c_0}  \overline{\psi}\phi
       + \frac{\rho_0}{c_0^2}(\mathbf{u}_0\cdot\mathbf{n})(\mathbf{u}_0\cdot
       \boldsymbol{\tau})\overline{\psi}\frac{\partial\phi}{\partial n}
       \,\mathrm{d}\Gamma \;,

    where :math:`\psi` is the test function.
    The following term is added to the right-hand side of the system:

    .. math::

       -\int_\Gamma \rho_0 \left[ 1 - (\mathbf{u}_0\cdot\mathbf{n}/c_0)^2\right]
       \overline{\psi} g \,\mathrm{d}\Gamma \;,

    In the following code snippet we apply this boundary condition with a constant source term
    :math:`g=1` on the boundary defined as the mesh group 3:

    .. code-block:: python

        from pfe.models.lpe_2d import CBC

        g = Constant(1.0)
        model.add_term(CBC(mesh.group(3)), g)

    A complete example is provided in the Jupyter notebook ``examples/lpe_2d/cbc``.
    """

    def __init__(self, domain, source=Constant(0.0)):
        """Constructor

        :param domain: The finite-element domain
        :type domain: An instance of pfe.Model
        :param source: A source function, defaults to Constant(0.0)
        :type source: An instance of pfe.Constant, pfe.Function or an interpolated field, optional
        """
        self.domain = domain
        self.source = source

    def assemble(self, model, system):
        """Assemble the element matrices

        :param model: The finite-element model
        :type model: An instance of pfe.Model
        :param system: The algebraic system to contribute to
        :type system: An instance of a class from pfe.algebra
        """
        elements = self.domain.get_elements(dim=1)
        for element in elements:
            Ke, Fe = self.terms(model, element)
            dof = model.fields["phi"].element_dofs(element)
            i = np.repeat(dof, len(dof))
            j = np.tile(dof, len(dof))
            system.lhs.add(i, j, Ke.flatten())
            system.rhs.add(dof, dof * 0, Fe.flatten())

    def terms(self, model, e):
        """Compute the element matrices for a single element

        :param model: The finite-element model
        :type model: An instance of pfe.Model
        :param e: The element tag
        :type e: A positive integer
        :return: The element matrices
        :rtype: A tuple of Numpy arrays
        """
        basis = model.fields["phi"].basis(e)
        geometry = self.domain.element_geometry(e)
        quad_order = 2 * basis.order
        u, weights = geometry.integration(quad_order)
        xy = geometry.position(u)
        tau = geometry.tangent(u)
        n = geometry.normal(u)
        phi, dphidtau = geometry.basis_from_order(basis, quad_order)
        omega = model.parameters["omega"].get_value()
        u0 = model.parameters["u0"].get_value(e, u, xy)
        v0 = model.parameters["v0"].get_value(e, u, xy)
        rho0 = model.parameters["rho0"].get_value(e, u, xy)
        c0 = model.parameters["c0"].get_value(e, u, xy)
        g = self.source.get_value(e, u, xy)
        u0tau = u0 * tau[:, 0] + v0 * tau[:, 1]
        u0n = u0 * n[:, 0] + v0 * n[:, 1]
        Ke = (
            -phi.T @ np.diag(weights * rho0 * u0n * u0tau / c0 ** 2) @ dphidtau
            - phi.T @ np.diag(weights * rho0 * 1j * omega / c0) @ phi
        )
        Fe = -phi.T @ np.diag(weights * rho0 * (1 - (u0n / c0) ** 2)) @ g
        return (Ke, Fe)
