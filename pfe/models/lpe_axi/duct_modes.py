"""Duct mode boundary condition for the LPE in cylindrical coordinates"""
import numpy as np
from numpy.lib.scimath import sqrt
from scipy.spatial.distance import pdist, squareform
from scipy.special import jv, jvp, jnp_zeros


class Basis:
    """Modal basis for a circular duct"""

    def __init__(self):
        """Constructor"""
        self.num_modes = None
        self.c0 = None
        self.u0n = None
        self.normal = None
        self.xc = None
        self.Ri = None
        self.Ro = None
        self.omega = None
        self.mode_m = None

    def prepare(self):
        if self.Ri == 0.0:
            self.kr = jnp_zeros(self.mode_m, self.num_modes)/self.Ro
            if self.mode_m == 0:
                self.kr[1:] = self.kr[:-1]
                self.kr[0] = 0.0

    def phi(self, _, y):
        """Mode shape functions

        :param x: Physical coordinate
        :type x: A scalar or a Numpy array
        :param y: Physical coordinate
        :type y: A scalar or a Numpy array
        :return: Mode shape functions in the two propagation directions
        :rtype: A tuple of two Numpy arrays
        """
        if self.Ri == 0.0:
            psi = jv(self.mode_m, np.outer(y, self.kr))
        return (psi, psi)

    def dphidn(self, _, y):
        """Normal derivatives of the mode shape functions

        :param x: Physical coordinate
        :type x: A scalar or a Numpy array
        :param y: Physical coordinate
        :type y: A scalar or a Numpy array
        :param omega: Angular frequency
        :type omega: A scalar
        :return: Mode shape functions in the two propagation directions
        :rtype: A tuple of two Numpy arrays
        """
        k_o = np.conj(
            (
                self.c0 * sqrt(self.omega ** 2 - (self.c0 ** 2 - self.u0n ** 2) * self.kr ** 2)
                - self.u0n * self.omega
            )
            / (self.c0 ** 2 - self.u0n ** 2)
        )
        k_i = np.conj(
            (
                -self.c0
                * sqrt(self.omega ** 2 - (self.c0 ** 2 - self.u0n ** 2) * self.kr ** 2)
                - self.u0n * self.omega
            )
            / (self.c0 ** 2 - self.u0n ** 2)
        )
        if self.Ri == 0.0:
            psi = jv(self.mode_m, np.outer(y, self.kr))
        psi_o = -1j * k_o[None, :] * psi
        psi_i = -1j * k_i[None, :] * psi
        return (psi_o, psi_i)

    def dphidtau(self, _, y):
        """Tangential derivatives of the mode shape functions

        :param x: Physical coordinate
        :type x: A scalar or a Numpy array
        :param y: Physical coordinate
        :type y: A scalar or a Numpy array
        :return: Mode shape functions in the two propagation directions
        :rtype: A tuple of two Numpy arrays
        """
        if self.Ri == 0.0:
            psi = self.kr[None,:]*jvp(self.mode_m, np.outer(y, self.kr))
        return (psi, psi)

    def norms(self):
        """Norms of the mode shape functions

        :return: The norms of the outgoing and incoming mode shape functions
        :rtype: A tuple of two Numpy arrays
        """
        Q = np.ones((self.num_modes,))
        if self.Ri==0:
            for n in range(self.num_modes):
                kr = self.kr[n]
                if kr==0:
                    Q[n] = self.Ro**2/2
                else:
                    Q[n] = (1-self.mode_m**2/(kr*self.Ro)**2)*self.Ro**2*jv(self.mode_m, kr*self.Ro)**2/2
        N = np.diag(2*np.pi*Q)
        return (N, N)


class DuctModes:
    """Duct mode boundary condition for the LPE in 2D"""

    def __init__(self, domain, modes_o, modes_i=None):
        """Constructor

        :param domain: The finite-element model
        :type domain: An instance of pfe.Model
        :param Ao: Amplitude of the outgoing modes
        :type Ao: An instance of pfe.Vector
        :param Ai: Amplitude of the incoming modes, defaults to None
        :type Ai: A sequence of scalars, optional
        """
        self.domain = domain
        self.modes_o = modes_o
        if modes_i is not None:
            self.modes_i = np.asarray(modes_i)
        else:
            self.modes_i = None
        self.basis = Basis()
        # Get the normal vector from the first element
        e = self.domain.get_elements(dim=1)[0]
        self.basis.normal = self.domain.element_geometry(e).normal(np.array([[0.0]]))[0]
        # Compute the width and center of the boundary
        # We assume that the surface is flat
        nodes = self.domain.nodes()
        self.basis.Ri = np.min(nodes[1,:])
        self.basis.Ro = np.max(nodes[1,:])
        D = squareform(pdist(nodes.T))
        n1, n2 = np.unravel_index(np.argmax(D), D.shape)
        self.basis.xc = (nodes[:, n1] + nodes[:, n2]) / 2
        print(self.basis.xc)

    def assemble(self, model, system):
        """Assemble the element matrices

        :param model: The finite-element model
        :type model: An instance of pfe.Model
        :param system: The algebraic system ton contribute to
        :type system: An instance of a class from pfe.algebra
        """
        elements = self.domain.get_elements(dim=1)
        # Set the number of modes to use
        self.basis.num_modes = model.fields[self.modes_o].length
        # Compute the mean sound speed and flow normal velocity
        u0fun = model.parameters["u0"]
        v0fun = model.parameters["v0"]
        c0fun = model.parameters["c0"]
        quad_order = u0fun.order
        u0 = 0.0
        v0 = 0.0
        c0 = 0.0
        L = 0.0
        for element in elements:
            geometry = self.domain.element_geometry(element)
            u, weights = geometry.integration(quad_order)
            u0 += weights.dot(u0fun.get_value(element, u))
            v0 += weights.dot(v0fun.get_value(element, u))
            c0 += weights.dot(c0fun.get_value(element, u))
            L += np.sum(weights)
        self.basis.c0 = c0 / L
        self.basis.u0n = (u0 * self.basis.normal[0] + v0 * self.basis.normal[1]) / L
        # Angular frequency
        self.basis.omega = model.parameters["omega"].get_value()
        # Azimuthal order
        self.basis.mode_m = model.parameters["m"].get_value()
        # Compute radial wavenumbers
        self.basis.prepare()
        # Assemble the element contributions
        for element in elements:
            K12, K21, F1 = self.terms(model, element)
            dof1 = model.fields["phi"].element_dofs(element)
            dof2 = model.fields[self.modes_o].dofs
            system.lhs.add_matrix(dof1, dof2, K12)
            system.lhs.add_matrix(dof2, dof1, K21)
            if F1 is not None:
                system.rhs.add(dof1, dof1 * 0, F1.flatten())
        Q_oo, Q_io = self.basis.norms()
        K22 = Q_oo.copy()
        system.lhs.add_matrix(dof2, dof2, K22)
        if self.modes_i is not None:
            F2 = -Q_io @ self.modes_i
            system.rhs.add(dof2, dof2 * 0, F2.flatten())

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
        quad_order = basis.order + 3
        u, weights = geometry.integration(quad_order)
        xy = geometry.position(u)
        x, y = xy
        weights *= 2*np.pi*y
        tau = geometry.tangent(u)
        n = geometry.normal(u)
        phi, _ = geometry.basis_from_order(basis, quad_order)
        omega = model.parameters["omega"].get_value()
        u0 = model.parameters["u0"].get_value(e, u, xy)
        v0 = model.parameters["v0"].get_value(e, u, xy)
        rho0 = model.parameters["rho0"].get_value(e, u, xy)
        c0 = model.parameters["c0"].get_value(e, u, xy)
        u0tau = u0 * tau[:, 0] + v0 * tau[:, 1]
        u0n = u0 * n[:, 0] + v0 * n[:, 1]
        phi_o, phi_i = self.basis.phi(x, y)
        dphidn_o, dphidn_i = self.basis.dphidn(x, y)
        dphidtau_o, dphidtau_i = self.basis.dphidtau(x, y)
        if self.modes_i is not None:
            F1 = (
                phi.T @ ((1j * omega * weights * rho0 * u0n / c0 ** 2)[:, None] * phi_i)
                + phi.T
                @ ((weights * rho0 * u0n * u0tau / c0 ** 2)[:, None] * dphidtau_i)
                - phi.T @ ((weights * rho0 * (1 - (u0n / c0) ** 2))[:, None] * dphidn_i)
            ) @ self.modes_i
        else:
            F1 = None
        K12 = (
            -phi.T @ ((1j * omega * weights * rho0 * u0n / c0 ** 2)[:, None] * phi_o)
            - phi.T @ ((weights * rho0 * u0n * u0tau / c0 ** 2)[:, None] * dphidtau_o)
            + phi.T @ ((weights * rho0 * (1 - (u0n / c0) ** 2))[:, None] * dphidn_o)
        )
        K21 = -phi_o.T @ (weights[:, None] * phi)
        return (K12, K21, F1)
