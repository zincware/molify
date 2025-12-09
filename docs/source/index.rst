molify Documentation
====================

**molify** - molecular structure interface with RDKit, ASE, Packmol and NetworkX

The molify package provides tools to convert molecular structures between
RDKit_, ASE_ and NetworkX_.
Furthermore, it provides a python interface for building periodic simulation boxes using Packmol_.

- RDKit is a versatile and popular cheminformatics library.

- ASE allows interfacing with a wide range of atomistic simulation codes and reads/writes many file formats.

- NetworkX is designed for the creation, manipulation, and study of complex networks.

- Packmol is a tool for building initial configurations for molecular dynamics simulations.

Key Features
------------

- **Bidirectional conversions** between RDKit Mol, ASE Atoms, and NetworkX Graph objects
- **Connectivity preservation** with automatic bond detection when needed
- **Molecular system building** from SMILES to solvated systems via an additional Packmol_ interface

Conversion Overview
-------------------

These bridging functions are illustrated below.

.. mermaid::

   graph LR
      A[rdkit.Mol] -->|molify.rdkit2networkx| B[networkx.Graph]
      B -->|molify.networkx2ase| C[ase.Atoms]
      A -->|molify.rdkit2ase| C

.. mermaid::

   graph LR
      A[ase.Atoms] -->|molify.ase2networkx| B[networkx.Graph]
      B -->|molify.networkx2rdkit| C[rdkit.Mol]
      A -->|molify.ase2rdkit| C

Conversion Details
------------------
The rdkit package is built around the concept of molecules, with defined connections, bond orders and charges. Positional information is optional.
In contrast, ASE Atoms objects are built around atomic positions and atomic numbers, with no inherent concept of bonds or connectivity.
NetworkX Graphs are general-purpose graph structures with no predefined chemistry concepts.

To store the connectivity information inside ASE Atoms objects, molify introduces a custom attribute ``ase.Atoms.info['connectivity']``: a list of tuples ``(atom_index_1, atom_index_2, bond_order)`` where indices are 0-based integers and ``bond_order`` can be a float following RDKit convention or None for unknown bond orders.

When connectivity information is not available:

- **ase2networkx**: Uses covalent radii to estimate bonds between atoms.
- **networkx2rdkit**: Uses ``rdkit.Chem.rdDetermineBonds.DetermineBonds`` to determine bond orders from 3D coordinates.


Quick Start
-----------

**Basic Conversions**

We can convert any RDKit molecule to an ASE Atoms object:

.. code-block:: python

   import molify
   from rdkit import Chem

   etoh = Chem.MolFromSmiles("CCO")
   etoh = Chem.AddHs(etoh)

   atoms = molify.rdkit2ase(etoh)


Likewise, we can convert the ASE Atoms object back to an RDKit molecule:

.. code-block:: python

   import molify
   import ase.build

   nh3 = ase.build.molecule("NH3")
   mol = molify.ase2rdkit(nh3)

**NetworkX Conversions**

The networkx.Graph representation can be obtained from either RDKit or ASE objects:

.. code-block:: python

   import molify
   import ase.build
   from rdkit import Chem

   etoh_mol = Chem.MolFromSmiles("CCO")
   nh3_atoms = ase.build.molecule("NH3")

   etoh_graph = molify.rdkit2networkx(etoh_mol)
   nh3_graph = molify.ase2networkx(nh3_atoms)


Advanced Features
-----------------

**3D Conformer Generation**

The main functionality of molify is expanded by additional tools that combine multiple packages.
For example, molify provides helper functions to generate 3D conformers directly from SMILES strings and store them as ASE Atoms objects.

.. code-block:: python

   import molify

   # Generate 10 different 3D conformers of water
   water = molify.smiles2conformers("O", numConfs=10)


**Molecular System Building**

For many applications, it is useful to solvate molecules or create mixtures.
molify provides a convenient interface to Packmol_ for building molecular systems.

.. code-block:: python

   import molify

   # Generate conformers for water and ethanol
   water = molify.smiles2conformers("O", numConfs=10)
   etoh = molify.smiles2conformers("CCO", numConfs=10)

   # Pack 5 water and 5 ethanol molecules into a box at 800 kg/m³
   box = molify.pack(
      data=[water, etoh], counts=[5, 5], density=800
   )

More details on these tools can be found in the :doc:`packmol_tools`, :doc:`ase_tools`, :doc:`rdkit_tools`, and :doc:`atom_selection` sections.


Installation
------------

**From PyPI**

You can install molify from PyPI:

.. code-block:: console

   (.venv) $ pip install molify

**From Source**

To install and develop molify from source, we recommend using `uv <https://docs.astral.sh/uv>`_.

More information and installation instructions can be found in the `UV documentation <https://docs.astral.sh/uv/getting-started/installation/>`_.

.. code-block:: console

   (.venv) $ git clone https://github.com/zincware/molify
   (.venv) $ cd molify
   (.venv) $ uv sync
   (.venv) $ source .venv/bin/activate

.. toctree::
   :maxdepth: 2
   :hidden:

   rdkit_tools
   ase_tools
   packmol_tools
   networkx_tools
   atom_selection
   modules


.. _RDKit: https://www.rdkit.org/
.. _ASE: https://wiki.fysik.dtu.dk/ase/
.. _Packmol: https://github.com/m3g/Packmol
.. _NetworkX: https://networkx.org/
