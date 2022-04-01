import numpy as np
import pytest
import os


params = "freq, Mach, theta, alpha"
cases = [(1.0, 0.0, 0.0, 0.0),
         (2.1, 0.0, 0.0, 0.0),
         (3.1, 0.0, 0.0, 0.0),
         (3.7, 0.0, 0.0, 0.0),
         (2.1, 0.2, 1.0, 2.0),
         (3.1, 0.2, -1.0, 2.0),
         (3.7, 0.2, 3.0, 3.0),
         (2.1, 0.5, 0.0, np.pi),
         (3.1, 0.5, 1.0, 0.5),
         (3.7, 0.5, 3.0, 0.1)]

@pytest.mark.parametrize(params, cases)
def test_cbc(freq, Mach, theta, alpha):
    from pfe import Mesh, Model, Constant, Function
    from pfe.interpolation import Lagrange2
    from pfe.models import lpe_2d

    omega = 2*np.pi*freq
    rho0 = 1.0
    c0 = 1.0
    h = 0.03  # Element size

    k = omega/c0/(1+Mach*np.cos(theta-alpha))
    kx = k*np.cos(theta)
    ky = k*np.sin(theta)

    def g(xy):
        phi = np.exp(-1j*kx*xy[0,:]-1j*ky*xy[1,:])
        theta = np.arctan2(xy[1,:], xy[0,:])
        dphidn = -1j*(kx*np.cos(theta) + ky*np.sin(theta))*phi
        k = omega/c0/(1+Mach*np.cos(theta-alpha))
        return(dphidn+1j*k*phi)

    mesh = Mesh(os.path.join(os.path.dirname(__file__), "domain1.msh"), num_dim=2)

    model = Model()

    model.parameters['omega'] = Constant(omega)
    model.parameters['rho0'] = Constant(rho0)
    model.parameters['c0'] = Constant(c0)
    model.parameters['u0'] = Constant(Mach*c0*np.cos(alpha))
    model.parameters['v0'] = Constant(Mach*c0*np.sin(alpha))

    model.fields['phi'] = Lagrange2(mesh)

    model.add_term(lpe_2d.Main(mesh.group(0)))
    model.add_term(lpe_2d.CBC(mesh.group(1), Function(g)))

    model.declare_fields()
    model.build()
    model.solve()

    x = mesh.coordinates[0,:]
    y = mesh.coordinates[1,:]
    phi_ref = np.exp(-1j*kx*x-1j*ky*y)

    # Relative L2 error
    error = np.sqrt(np.sum(np.abs(model.solution-phi_ref[model.fields['phi'].dofs])**2)/np.sum(np.abs(phi_ref[model.fields['phi'].dofs])**2))
    # Interpolation order
    P = 2
    # Error estimator for the L2 error
    Ed = (1-Mach)/2*(np.math.factorial(P)/np.math.factorial(2*P))**2/(2*P+1)*(k*h)**(2*P)*k

    assert error < 2*Ed, "Numerical error is too high"
