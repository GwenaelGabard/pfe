import numpy as np
import matplotlib.pyplot as plt
import pytest
import os
from numpy.lib.scimath import sqrt


params = "freq, Z"
cases = [(2, 1), (2, 0.5-1j), (3.2, 1-0.5j), (3.2, 1j)]

@pytest.mark.parametrize(params, cases)
def test_impedance(freq, Z):
    from pfe import Mesh, Model, Constant
    from pfe.interpolation import Lagrange2
    from pfe.models import lpe_2d

    rho0 = 1.0
    c0 = 1.0
    u0 = 0.0
    L = 2
    tolerance = 2.e-6

    omega = 2*np.pi*freq

    mesh = Mesh(os.path.join(os.path.dirname(__file__), "duct.msh"), num_dim=2)

    model = Model()

    model.parameters['omega'] = Constant(omega)
    model.parameters['rho0'] = Constant(rho0)
    model.parameters['c0'] = Constant(c0)
    model.parameters['u0'] = Constant(u0)
    model.parameters['v0'] = Constant(0.0)

    model.fields['phi'] = Lagrange2(mesh)

    model.terms.append(lpe_2d.Main(mesh.group(0)))
    model.terms.append(lpe_2d.Wall(mesh.group([1, 3])))
    model.terms.append(lpe_2d.CBC(mesh.group(4), Constant(1)))
    model.terms.append(lpe_2d.Impedance(mesh.group(2), Constant(Z)))

    model.declare_fields()
    model.build()
    model.solve()

    kp = omega/(u0+c0)
    km = omega/(u0-c0)
    A = -1j/(kp-km)
    B = A*(kp-omega*rho0/Z)*np.exp(-1j*(kp-km)*L)/(omega*rho0/Z-km)

    x = mesh.coordinates[0,:]
    phi_ref = A*np.exp(-1j*kp*x) + B*np.exp(-1j*km*x)

    assert np.max(np.abs(model.solution-phi_ref[model.fields['phi'].dofs])) < tolerance
