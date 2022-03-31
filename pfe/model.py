"""The finite-element model"""
import time
from pfe.algebra import LinearSystem


class Model:
    """
    Model
    =====

    This class gathers all the ingredients describing the finite-element model.
    The definition and resolution of a model are controlled mainly through an instance of this
    class.
    It includes the following data structure.

    Parameters
    ----------

    This includes the parameters of the physical problem (like the frequency) and the
    known coefficients of the differential equations (like a non-uniform sound speed) involved in
    the problem.

    This information is stored in a dictionnary called ``parameters`` where the keys are the names
    of the parameters, and the values are instances of classes such as Vector, Constant, Function
    or Lagrange2.

    For instance to define a constant angular frequency for an acoustic problem we can use:
    """

    def __init__(self):
        """Constructor"""
        self.parameters = {}
        self.fields = {}
        self.terms = []
        self.system = LinearSystem()
        self.solution = None

    def declare_fields(self):
        """Declare the degrees of freedom of the fields to the algebraic system"""
        self.system.clear()
        print("")
        print("Allocating fields...")
        total = 0
        for name, field in self.fields.items():
            field.declare_dofs(self.system)
            print("* Field " "{}" ": {} DOFs".format(name, field.num_dofs()))
            total += field.num_dofs()
        print("Total: {} DOFs".format(total))

    def add_term(self, term):
        """Add a term to the variational formulation

        :param term: The term to add
        :type term: An instance of a class defined in one of the pfe models
            (e.g. Helmholtz2D.Velocity)
        """
        self.terms.append(term)

    def build(self):
        """Build the algebraic systems based on the terms declared in the variational
        formulation
        """
        print("")
        print("Building model...")
        tic = time.perf_counter()
        for term in self.terms:
            print("* Group {}".format(type(term).__name__))
            term.assemble(self, self.system)
        self.system.assemble()
        toc = time.perf_counter()
        print(f"Time: {toc - tic:0.4f}s")
        print(
            "Global system: {} DOFs, {} non-zero entries".format(
                self.system.num_dofs, self.system.lhs.num()
            )
        )

    def solve(self):
        """Solve the algebraic system and store the solution"""
        print("")
        print("Solving model...")
        tic = time.perf_counter()
        self.solution = self.system.solve()
        toc = time.perf_counter()
        print(f"Time: {toc - tic:0.4f}s")
