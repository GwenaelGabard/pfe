"""Myers impedance condition for the Helmholtz equation in 2D"""
import numpy as np


class Impedance:
    r"""Implements the impedance boundary condition for the Helmholtz equation in 2D:

    .. math::

       \frac{\partial p}{\partial n}=-\mathrm{i}\omega\rho_0 \frac{p}{Z} \;,

    where :math:`Z` is the acoustic impedance of the boundary :math:`\Gamma`.

    To implement this boundary condition, the following term is added to the left-hand side of the
    linear system:

    .. math::

       -\int_\Gamma \frac{\mathrm{i}\omega\rho_0}{Z} qp \,\mathrm{d}\Gamma \;,

    where :math:`q` is the test function.

    In the following code snippet we apply this boundary condition with a constant impedance
    :math:`1-\mathrm{i}/2` on the boundary defined as the mesh group 2:

    .. code-block:: python

        from pfe.models.helmholtz_2d import Impedance

        Z = Constant(1.0-0.5j)
        model.add_term(Impedance(mesh.group(2)), Z)

    A complete example is provided in the Jupyter notebook ``examples/helmholtz_2d/impedance``.
    """

    def __init__(self, domain, impedance):
        """Constructor

        :param domain: The mesh on which the terms are computed
        :type domain: An instance of pfe.Mesh
        :param impedance: The acoustic impedance of the surface
        :type impedance: An instance of pfe.Constant, pfe.Function or an interpolated field
        """
        self.domain = domain
        self.Z = impedance

    def assemble(self, model, system):
        """Assemble the element matrices

        :param model: The finite-element model
        :type model: An instance of pfe.Model
        :param system: The algebraic system to contribute to
        :type system: An instance of a class from pfe.algebra
        """
        elements = self.domain.get_elements(dim=1)
        for element in elements:
            Ke = self.terms(model, element)
            dof = model.fields["phi"].element_dofs(element)
            i = np.repeat(dof, len(dof))
            j = np.tile(dof, len(dof))
            system.lhs.add(i, j, Ke.flatten())

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
        quad_order = 2 * basis.order
        u, weights = geometry.integration(quad_order)
        xy = geometry.position(u)
        tau = geometry.tangent(u)
        phi, dphidtau = geometry.basis_from_order(basis, quad_order)
        omega = model.parameters["omega"].get_value()
        u0 = model.parameters["u0"].get_value(e, u, xy)
        v0 = model.parameters["v0"].get_value(e, u, xy)
        rho0 = model.parameters["rho0"].get_value(e, u, xy)
        Z = self.Z.get_value(e, u, xy)
        u0tau = u0 * tau[:, 0] + v0 * tau[:, 1]
        D0phiDt = 1j * omega * phi + np.diag(u0tau) @ dphidtau
        Ke = D0phiDt.T.conj() @ np.diag(weights * rho0**2 / Z / 1j / omega) @ D0phiDt
        return Ke
