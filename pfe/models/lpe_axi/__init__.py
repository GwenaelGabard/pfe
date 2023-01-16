r"""
Linearized Potential Equation (AXI)
===================================

The module ``lpe_axi`` provides an implementation of the Linearised Potential 
Equation (LPE) in cylindrical coordinates.

Theory
------

This equation describes the propagation of linear sound waves in a steady, 
potential base flow.
The sound field is described by the velocity potential :math:`\phi`.
The base flow is defined by the mean density :math:`\rho_0(\mathbf{x})`, the 
sound speed
:math:`c_0(\mathbf{x})` and the mean velocity :math:`\mathbf{u}_0(\mathbf{x})`.
We therefore assume that the mean flow velocity satisfies
:math:`\nabla\times\mathbf{u}_0=\mathbf{0}`.
The Linearised Potential Equation (LPE) reads

.. math::
    \rho_0 \frac{\mathrm{D}_0}{\mathrm{D}t}
    \left( \frac{1}{c_0^2}\frac{\mathrm{D}_0\phi}{\mathrm{D}t} \right) -
    \nabla \cdot ( \rho_0 \nabla \phi ) = 0\;,

where :math:`\mathrm{D}_0\phi/\mathrm{D}t=\partial\phi/\partial t +
\mathbf{u}_0\cdot\nabla\phi` is the material derivative with respect to the 
base flow.
This model is solved in the frequency domain using the implicit time dependence
:math:`\exp(+\mathrm{i}\omega t)`.
We therefore have :math:`\mathrm{D}_0\phi/\mathrm{D}t=\mathrm{i}\omega\phi +
\mathbf{u}_0\cdot\nabla\phi`.
In cylindrical coordinates, the velocity potential is of the form

.. math::
    \phi(x,r,\theta) = \phi(x,r)\exp(-\mathrm{i}m\theta)\;,

where :math:`m` is the azimuthal order.

This propagation model is a simplified version of the theory developped by 
Goldstein :cite:`goldstein78` without the entropy and vortical disturbances.

From the knowledge of the velocity potential :math:`\phi`, one can recover 
other acoustic quantities:

* The acoustic velocity :math:`\mathbf{u}=\nabla\phi`.
* The acoustic pressure :math:`p=-\rho_0\mathrm{D}_0\phi/\mathrm{D}t`.

The LPE is formulated as a variational statement:

.. math::
    \int_\Omega \frac{\rho_0}{c_0^2} \overline{\frac{\mathrm{D}_0\psi}{\mathrm{D}t}}
    \frac{\mathrm{D}_0\phi}{\mathrm{D}t}
    - \rho_0\nabla\overline{\psi}\cdot\nabla\psi \,\mathrm{d}\Omega
    +\int_{\partial\Omega} \rho_0\overline{\psi}\frac{\partial\phi}{\partial n}
    -\frac{\rho_0}{c_0^2} (\mathbf{u}_0\cdot\mathbf{n}) \overline{\psi}
    \frac{\mathrm{D}_0\phi}{\mathrm{D}t}\,\mathrm{d}\Gamma
    =0\;,

for any test function :math:`\psi`.
The boundary terms are then modified to implement various boundary conditions.
These are described below in details.

Implementation
--------------

This formulation requires the definition of the following quantities as 
parameters in the model
(i.e. in the dictionary ``model.parameters``):

==========================================================  ==============  =================
Description                                                 Parameter name  Type
==========================================================  ==============  =================
Angular frequency :math:`\omega`                            ``omega``       Constant
Azimuthal order :math:`m`                                   ``m``        Constant integer
Mean density :math:`\rho_0`                                 ``rho0``        Constant or field
Sound speed :math:`c_0`                                     ``c0``          Constant or field
:math:`x`-coordinate of the mean flow velocity :math:`u_0`  ``u0``          Constant or field
:math:`r`-coordinate of the mean flow velocity :math:`v_0`  ``v0``          Constant or field
==========================================================  ==============  =================

In addition a field called ``phi`` should be defined in the dictionnary ``model.fields``
to represent the velocity potential :math:`\phi`.

The following is an example of usage of this sub-module:

.. code-block:: python

   from pfe import Model, Constant
   from pfe.interpolation import Lagrange2

   model = Model()
   model.parameters['omega'] = Constant(omega)
   model.parameters['m'] = Constant(2)
   model.parameters['rho0'] = Constant(1.2)
   model.parameters['c0'] = Constant(340)
   model.parameters['u0'] = Constant(100)
   model.parameters['v0'] = Constant(0)

   model.fields['phi'] = Lagrange2(mesh)

Main terms
~~~~~~~~~~

.. automodule:: pfe.models.lpe_axi.Main
   :members:

Imposed normal velocity
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfe.models.lpe_axi.Velocity
   :members:

Hard wall
~~~~~~~~~

.. automodule:: pfe.models.lpe_axi.Wall
   :members:

Acoustic impedance 
~~~~~~~~~~~~~~~~~~

.. automodule:: pfe.models.lpe_axi.Impedance
   :members:

Duct modes boundary
~~~~~~~~~~~~~~~~~~~

.. automodule:: pfe.models.lpe_axi.DuctModes
   :members:

References
----------
.. bibliography:: references.bib

"""

from .main import Main
from .wall import Wall
from .impedance import Impedance
from .velocity import Velocity
from .duct_modes import DuctModes
from .axis import Axis

__all__ = [
    "Main",
    "Wall",
    "Impedance",
    "Velocity",
    "DuctModes",
    "Impedance",
    "Axis",
]
