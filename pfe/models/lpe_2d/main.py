"""Main terms for the LPE in 2D"""
import numpy as np


class Main:
    r"""Implements the main terms for the Linearised Potential Equation in 2D.
    The following terms are added to the left-hand side of the linear system:

    .. math::

       \int_\Omega \frac{\rho_0}{c_0^2} \overline{\frac{\mathrm{D}_0\psi}{\mathrm{D}t}}
       \frac{\mathrm{D}_0\phi}{\mathrm{D}t}
       - \rho_0\nabla\overline{\psi}\cdot\nabla\psi \,\mathrm{d}\Omega \;,

    where :math:`\phi` is the unknown field and :math:`\psi` is the associated test function.

    In the following code snippet we define these terms on the domain defined as the mesh group 0:

    .. code-block:: python

        from pfe.models.lpe_2d import Main

        model.add_term(Main(mesh.group(0)))

    Several complete examples are provided as Jupyter notebooks in ``examples/lpe_2d``.
    """

    def __init__(self, domain):
        """Constructor

        :param domain: The finite-element model
        :type domain: An instance of pfe.Model
        """
        self.domain = domain

    def assemble(self, model, system):
        """Assemble the element matrices

        :param model: The finite-element model
        :type model: An instance of pfe.Model
        :param system: The algebraic system ton contribute to
        :type system: An instance of a class from pfe.algebra
        """
        elements = self.domain.get_elements(dim=2)
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
        order = 2 * basis.order
        uv, weights = geometry.ref_integration(order)
        xy = geometry.position(uv)
        dxydu, dxydv, detJ = geometry.metric_from_order(order)
        weights *= detJ
        phi, dphidu, dphidv = basis.from_order(order)
        dphidx, dphidy = geometry.du_to_dx(dphidu, dphidv, dxydu, dxydv, detJ)

        omega = model.parameters["omega"].get_value()
        u0 = model.parameters["u0"].get_value(e, uv, xy)
        v0 = model.parameters["v0"].get_value(e, uv, xy)
        c0 = model.parameters["c0"].get_value(e, uv, xy)
        rho0 = model.parameters["rho0"].get_value(e, uv, xy)

        D0phiDt = 1j * omega * phi + u0[:, None] * dphidx + v0[:, None] * dphidy
        Ke = (
            D0phiDt.T.conj() @ ((weights * rho0 / c0 ** 2)[:, None] * D0phiDt)
            - dphidx.T @ ((weights * rho0)[:, None] * dphidx)
            - dphidy.T @ ((weights * rho0)[:, None] * dphidy)
        )
        return Ke
