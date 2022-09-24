"""Hard-wall boundary condition for the LPE in cylindrical coordinates"""


class Wall:
    r"""Implements the hard-wall boundary condition for the Linearised Potential
    Equation in cylindrical coordinates:

    .. math::

       \frac{\partial\phi}{\partial n}=0 \;.

    The mean flow is assumed to be tangential to the boundary:
    :math:`\mathbf{u}_0\cdot\mathbf{n}=0`.
    To implement this boundary condition, there is no term to add to the formulation.

    In the following code snippet we apply this boundary condition with a constant normal velocity
    on the boundary defined as the mesh group 3:

    .. code-block:: python

        from pfe.models import lpe_axi

        model.add_term(lpe_axi.Wall(mesh.group(3)))

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
        :param system: The algebraic system to contribute to
        :type system: An instance of a class from pfe.algebra
        """
