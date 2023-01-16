"""Boundary condition on the symmetry axis."""
import numpy as np


class Axis:
    """Boundary condition on the symmetry axis.

    If the azimuthal order m is different from 0, a Dirichlet condition is
    applied on the axis to force the potential to zero.
    """

    def __init__(self, domain):
        """Constructor for the axis boundary condition

        :param domain: Domain where the condition is applied
        :type domain: Mesh instance or sub-group of Mesh
        """
        self.domain = domain

    def assemble(self, model, system):
        """Add the linear constraint for the Dirichlet condition

        :param model: The finite-element model
        :type model: An instance of pfe.Model
        :param system: The algebraic system to contribute to
        :type system: An instance of a class from pfe.algebra
        """
        # When m=0, this is a homogeneous Neumann condition (natural condition)
        if model.parameters["m"].get_value() == 0:
            return
        # Constructing the list of dof for the potential on the boundary
        elements = self.domain.get_elements(dim=1)
        dofs = []
        for element in elements:
            dofs.append(model.fields["phi"].element_dofs(element))
        dofs = np.concatenate(dofs)
        dofs = np.unique(dofs)
        # Add a linear constraint for each dof
        for dof in dofs:
            system.add_constraint([dof], [1.0], 0.0)
