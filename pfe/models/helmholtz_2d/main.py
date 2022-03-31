"""Main terms for the Helmholtz equation in 2D"""
import numpy as np


class Main:
    r"""Implements the main terms for the Helmholtz equation in 2D.
    The following terms are added to the left-hand side of the linear system:

    .. math::

        \int_\Omega k^2qp - \boldsymbol{\nabla}q\cdot\boldsymbol{\nabla}p
        \,\mathrm{d}\Omega

    where :math:`p` is the unknown pressure field and :math:`q` is the associated test function.

    In the following code snippet we define these terms on the domain defined as the mesh group 0:

    .. code-block:: python

        from pfe.models.helmholtz_2d import Main

        model.add_term(Main(mesh.group(0)))

    Several complete examples are provided as Jupyter notebooks in ``examples/helmholtz_2d``.
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
            dof = model.fields["pressure"].element_dofs(element)
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
        basis = model.fields["pressure"].basis(e)
        geometry = self.domain.element_geometry(e)
        order = 2 * basis.order
        _, weights = geometry.ref_integration(order)
        dxydu, dxydv, detJ = geometry.metric_from_order(order)
        weights *= detJ
        p, dpdu, dpdv = basis.from_order(order)
        dpdx, dpdy = geometry.du_to_dx(dpdu, dpdv, dxydu, dxydv, detJ)

        omega = model.parameters["omega"].value
        c0 = model.parameters["c0"].value
        k = omega / c0

        Ke = (
            p.T() @ np.diag(weights * k ** 2) @ p
            - dpdx.T() @ np.diag(weights) @ dpdx
            - dpdy.T() @ np.diag(weights) @ dpdy
        )
        return Ke
