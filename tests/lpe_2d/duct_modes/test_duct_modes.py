import numpy as np
import matplotlib.pyplot as plt
import pytest
import os
from numpy.lib.scimath import sqrt


params = "mode_m, u0"
cases = [(0, 0.0), (1, 0.0), (0, 0.3), (1, 0.3), (0, -0.3), (1, -0.3)]
ids = ["no flow, mode 0", "no flow, mode 1", "downstream, mode 0", "downstream, mode 1", "upstream, mode 0", "upstream, mode 1"]

@pytest.mark.parametrize(params, cases, ids=ids)
def test_duct_modes(mode_m, u0):
    from pfe import Mesh, Model, Constant, Vector
    from pfe.interpolation import Lagrange2
    from pfe.models import lpe_2d

    freq = 1.2
    num_modes = 5
    H = 0.5
    rho0 = 1.0
    c0 = 1.0
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
    model.fields['R'] = Vector(num_modes)
    model.fields['T'] = Vector(num_modes)
    A_in = np.zeros((num_modes,))
    A_in[mode_m] = 1.0

    model.terms.append(lpe_2d.Main(mesh.group(0)))
    model.terms.append(lpe_2d.Wall(mesh.group([1, 3])))
    model.terms.append(lpe_2d.DuctModes(mesh.group(4), 'R', A_in))
    model.terms.append(lpe_2d.DuctModes(mesh.group(2), 'T'))

    model.declare_fields()
    model.build()
    model.solve()
    
    m = np.arange(num_modes)
    k_tau = m*np.pi/H
    k_o = np.conj((+c0*sqrt(omega**2-(c0**2-u0**2)*k_tau**2)-u0*omega)/(c0**2-u0**2))
    k_i = np.conj((-c0*sqrt(omega**2-(c0**2-u0**2)*k_tau**2)-u0*omega)/(c0**2-u0**2))

    T_ref = np.exp(-1j*k_o[mode_m]*L)

    Rv = np.zeros((num_modes,), dtype=complex)
    Tv = np.zeros((num_modes,), dtype=complex)
    Tv[mode_m] = T_ref*(-1)**mode_m

    ET = np.max(np.abs(Tv-model.solution[model.fields['T'].dofs]))
    ER = np.max(np.abs(Rv-model.solution[model.fields['R'].dofs]))

    assert (ET<tolerance) and (ER<tolerance)
