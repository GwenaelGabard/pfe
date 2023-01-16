"""Myers impedance condition for the LPE in cylindrical coordinates"""
import numpy as np


class Impedance:
    r"""Implements the Myers impedance boundary condition for the Linearised
    Potential Equation in cylindrical coordinates:

    .. math::

       \frac{\partial\phi}{\partial n} = \frac{\mathrm{D}_0}{\mathrm{D}t}
       \left( \frac{-\rho_0}{\mathrm{i}\omega Z}
       \frac{\mathrm{D}_0\phi}{\mathrm{D}t} \right) \;,

    where :math:`Z` is the acoustic impedance of the boundary :math:`\Gamma`.
    The mean flow is assumed to be tangential to the boundary:
    :math:`\mathbf{u}_0\cdot\mathbf{n}=0`.
    This boundary condition was proposed in :cite:`myers80` and assumes an
    infinitely thin boundary layer above the acoustic treatment.

    To implement this boundary condition, the following term is added to the
    left-hand side of the linear system:

    .. math::

       \int_\Gamma \frac{\rho_0^2}{\mathrm{i}\omega Z}
       \overline{\frac{\mathrm{D}_0\psi}{\mathrm{D}t}}
       \frac{\mathrm{D}_0\phi}{\mathrm{D}t} \,\mathrm{d}\Gamma \;,

    where :math:`\psi` is the test function.
    This formulation is the one proposed by Eversman :cite:`eversman01`.

    In the following code snippet we apply this boundary condition with a
    constant impedance :math:`1-\mathrm{i}/2` on the boundary defined as the
    mesh group 2:

    .. code-block:: python

        from pfe.models import lpe_axi

        Z = Constant(1.0-0.5j)
        model.add_term(lpe_axi.Impedance(mesh.group(2)), Z)

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
            system.lhs.add_matrix(dof, dof, Ke)

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
        quad_order = (
            8 * basis.order
        )  # to change to evaluate the impact of the quadrature order
        u, weights = geometry.integration(quad_order)
        xr = geometry.position(u)
        _, r = xr
        weights *= 2 * np.pi * r
        tau = geometry.tangent(u)
        phi, dphidtau = geometry.basis_from_order(basis, quad_order)
        omega = model.parameters["omega"].get_value()
        u0 = model.parameters["u0"].get_value(e, u, xr)
        v0 = model.parameters["v0"].get_value(e, u, xr)
        rho0 = model.parameters["rho0"].get_value(e, u, xr)
        inv_Z = 1 / self.Z.get_value(e, u, xr)
        u0tau = u0 * tau[:, 0] + v0 * tau[:, 1]
        D0phiDt = 1j * omega * phi + u0tau[:, None] * dphidtau
        Ke = D0phiDt.T.conj() @ (
            (weights * rho0**2 * inv_Z / 1j / omega)[:, None] * D0phiDt
        )
        return Ke
