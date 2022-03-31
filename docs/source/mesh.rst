Mesh
====

The class ``Mesh`` contains the data and methods to represent the finite element mesh.
It gathers primarily four types of data:

* Coordinates of the nodes (either in 1D, 2D or 3D)
* Definition of the elements (i.e. list of nodes for each element)
* Connections between elements (neighbouring elements of a given element)
* Groups of elements to define physical properties

.. autoclass:: pfe.Mesh
   :members:
