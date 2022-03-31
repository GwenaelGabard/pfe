r"""
Helmholtz equation (2D)
=======================

The module ``helmholtz_2d`` provides an implementation of the Helmholtz equation in two dimensions.

Theory
------

This equation describes the propagation of linear sound waves in a uniform, quiescent medium.
The sound field is described by the complex amplitude :math:`p` of the acoustic pressure.
The Helmholtz equation is written

.. math::
    k^2 p + \boldsymbol{\nabla}^2 p = 0 \;,

where :math:`k=\omega/c_0` is the acoustic wavenumber defined in terms of the angular frequency
:math:`\omega` and the sound speed :math:`c_0`.
This model is solved in the frequency domain using the implicit time dependence
:math:`\exp(+\mathrm{i}\omega t)`.

The Helmholtz equation is formulated as a variational statement:

.. math::
    \int_\Omega k^2qp - \boldsymbol{\nabla}q\cdot\boldsymbol{\nabla}p \,\mathrm{d}\Omega
    +\int_{\partial\Omega} q\frac{\partial p}{\partial n}\,\mathrm{d}\Gamma
    =0\;,

for any test function :math:`q`.
The boundary terms are then modified to implement various boundary conditions.
These are described below in details.

Implementation
--------------

This formulation requires the definition of the following quantities as parameters in the model
(i.e. in the dictionary ``model.parameters``):

==========================================================  ==============  =================
Description                                                 Parameter name  Type
==========================================================  ==============  =================
Angular frequency :math:`\omega`                            ``omega``       Constant
Mean density :math:`\rho_0`                                 ``rho0``        Constant
Sound speed :math:`c_0`                                     ``c0``          Constant
==========================================================  ==============  =================

In addition a field called ``pressure`` should be defined in the dictionnary ``model.fields``
to represent the velocity potential :math:`p`.

The following is an example of usage of this sub-module:

.. code-block:: python

   from pfe import Model, Constant
   from pfe.interpolation import Lagrange2

   model = Model()
   model.parameters['omega'] = Constant(omega)
   model.parameters['rho0'] = Constant(1.2)
   model.parameters['c0'] = Constant(340)

   model.fields['pressure'] = Lagrange2(mesh)

Main terms
~~~~~~~~~~

.. automodule:: pfe.models.helmholtz_2d.Main
   :members:

Imposed normal velocity
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfe.models.helmholtz_2d.Velocity
   :members:

Hard wall
~~~~~~~~~

.. automodule:: pfe.models.helmholtz_2d.Wall
   :members:

Acoustic impedance 
~~~~~~~~~~~~~~~~~~

.. automodule:: pfe.models.helmholtz_2d.Impedance
   :members:

Duct modes boundary
~~~~~~~~~~~~~~~~~~~

.. automodule:: pfe.models.helmholtz_2d.DuctModes
   :members:

Characteristic boundary condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfe.models.helmholtz_2d.CBC
   :members:

References
----------
.. bibliography:: references.bib
"""

from .main import Main
from .wall import Wall
from .cbc import CBC
from .impedance import Impedance
from .velocity import Velocity
from .duct_modes import DuctModes

__all__ = ["Main", "Wall", "CBC", "Impedance", "Velocity", "DuctModes"]
