"""Hard-wall boundary condition for the Helmholtz equation in 2D"""


class Wall:
    r"""Implements the hard-wall boundary condition for the Helmholtz equation in 2D:

    .. math::

       \frac{\partial p}{\partial n}=0 \;.

    To implement this boundary condition, there is no term to add to the formulation.

    In the following code snippet we apply this boundary condition with a constant normal velocity
    on the boundary defined as the mesh group 3:

    .. code-block:: python

        from pfe.models.helmholtz_2d import Wall

        model.add_term(Wall(mesh.group(3)))

    A complete example is provided in the Jupyter notebook ``examples/helmholtz_2d/hard_wall``.
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
