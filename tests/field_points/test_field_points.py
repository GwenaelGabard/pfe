import numpy as np
import pytest
import os


params = "order"
cases = [0, 1, 2, 3, 4]

@pytest.mark.parametrize(params, cases)
def test_field_points(order):
    from pfe import Mesh
    from pfe.interpolation import Lagrange2

    tolerance = 4.e-6

    mesh = Mesh(os.path.join(os.path.dirname(__file__), "domain.msh"), num_dim=2)

    field = Lagrange2(mesh)
    field.dofs = np.arange(mesh.num_nodes())

    xl = np.linspace(-1, 1, 200)
    line = mesh.locate_points(np.array([xl, xl*0]))

    x = mesh.coordinates[0,:]

    values = x**order

    f = field.sample(line, values)    
    f_ex = xl**order
    
    error = np.sqrt(np.sum(np.abs(f-f_ex)**2)/np.sum(np.abs(f_ex)**2))
    assert error < tolerance


@pytest.mark.parametrize(params, cases)
def test_field_point(order):
    from pfe import Mesh
    from pfe.interpolation import Lagrange2

    tolerance = 3.e-5

    mesh = Mesh(os.path.join(os.path.dirname(__file__), "domain.msh"), num_dim=2)

    field = Lagrange2(mesh)
    field.dofs = np.arange(mesh.num_nodes())

    xl = np.array([0.3])
    line = mesh.locate_points(np.array([xl, xl]))

    x = mesh.coordinates[0,:]

    values = x**order

    f = field.sample(line, values)    
    f_ex = xl**order
    
    error = np.sqrt(np.sum(np.abs(f-f_ex)**2)/np.sum(np.abs(f_ex)**2))
    print(f)
    print(f_ex)
    assert error < tolerance
