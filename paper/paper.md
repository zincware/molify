---
title: 'molify: Molecular Structure Interface'
tags:
  - Python
  - cheminformatics
  - MLIPs
  - ASE
  - RDKit
  - PACKMOL
  - NetworkX
authors:
  - name: Fabian Zills
    orcid: 0000-0002-6936-4692
    affiliation: "1"
affiliations:
 - name: Institute for Computational Physics, University of Stuttgart, 70569 Stuttgart, Germany
   index: 1
date: 2025-11-25
bibliography: bibliography.bib
---
# Summary

The increasing prevalence of Machine-Learned Interatomic Potentials (MLIPs) has shifted requirements for setting up atomistic simulations.
Unlike classical force fields, MLIPs primarily require atomic positions and species, thereby removing the need for predefined topology files used for classical force fields in molecular dynamics software like GROMACS [@abrahamGROMACSHighPerformance2015], LAMMPS [@LAMMPS], ESPResSo [@weikESPResSo40Extensible2019], or OpenMM [@eastmanOpenMM8Molecular2024].
Consequently, the Atomic Simulation Environment (ASE) [@larsenAtomicSimulationEnvironment2017] has become a popular Python toolkit for handling atomic structures and interfacing with MLIPs, particularly within the material science and soft matter communities.
ASE originates from the electronic structure community, which shares the same setup as MLIP-driven studies.
In contrast to _ab initio_, MLIPs are much faster and can run on much larger systems, making high-throughput simulations of more complex systems feasible and increasing the need for efficient initial structure generation.

Concurrently, RDKit [@landrumRdkitRdkit2023_03_22023] offers extensive functionality for cheminformatics and manipulating chemical structures.
However, standard RDKit workflows are not designed for MLIP-driven simulation, while typical ASE-MLIP workflows may lack rich explicit chemical information such as bond orders or molecular identities, as well as capabilities for generating different conformations or searching substructures.

The `molify` package bridges this gap, providing an interface between RDKit's chemical structure generation and cheminformatics capabilities and ASE's handling of 3D atomic structures.
Furthermore, `molify` integrates with PACKMOL [@martinezPACKMOLPackageBuilding2009] to facilitate the creation of complex, periodic simulation cells with diverse chemical compositions, all while preserving crucial chemical connectivity information.
In addition, `molify` simplifies the representation of molecular structures as graphs using NetworkX [@hagbergExploringNetworkStructure2008], e.g., enabling traversing or comparing them.
Lastly, the combination of these packages enables selection and manipulation of atomistic structures based on chemical knowledge rather than manual index handling.
While designed for MLIP data, the usage of `molify` is not limited and can be expanded, e.g., by utilizing the bond order information in other ASE-based workflows for classical MD simulations or integrating with machine-learning driven bond order predictions.


# Statement of need
`molify` serves as a vital link between RDKit, ASE, NetworkX, and PACKMOL, designed to aid working with molecular structures in systems without dedicated topology files and formats.
While its core function is to interface these tools, it thereby unlocks new capabilities and significantly reduces the manual coding and data wrangling typically required for preparing and analyzing molecular simulations.
For example, ASE has no tools for handling topological information such as bonds or molecular identities, while RDKit cannot natively interface with MLIPs.

`molify` simplifies workflows that previously involved laborious tasks such as sourcing individual structure files from various databases (e.g., the Materials Project [@jainCommentaryMaterialsProject2013] or the ZINC database [@tingleZINC22AFreeMultiBillionScale2023]) and custom setups of simulation cells.
With `molify`, more complex and chemically diverse simulation cells are easier to set up and process.

One challenge in MLIP-driven simulations is the post-simulation identification and analysis of molecular fragments or chemical changes, as explicit topological information is not available and changes in connectivity can occur.
`molify` addresses this by enabling the use of RDKit's powerful SMILES [@weiningerSMILESChemicalLanguage1988]/SMARTS-based substructure searching on ASE structures.
In addition, the resulting molecular graph can be exported to a NetworkX [@hagbergExploringNetworkStructure2008] object for further analysis.
This selection and handling allows for similar functionality as is provided by the MDAnalysis [@gowersMDAnalysisPythonPackage2016] atom selection language, designed for simulations with a fixed topology.

# Features and Implementation

![Visualization of a 3D structure from ASE, visualized with ZnDraw [@elijosiusZeroShotMolecular2024] (left) and its corresponding RDKit 2D chemical structure representation (right).\label{fig:zndraw-rdkit}](zndraw_rdkit.svg)

The generation of atomic configurations in `molify` is centered around SMILES for defining molecular species.
A typical workflow often follows these steps:

1. Generating 3D conformers for individual molecular species from their SMILES using RDKit.
2. Packing these conformers into a simulation box with a target density using PACKMOL and returning the simulation box as an ASE Atoms object.
3. Running MLIP-based simulations using ASE.
4. Post-processing the simulation data by identifying and selecting structures based on SMARTS.

```python
from molify import pack, smiles2conformers
from ase.optimize import LBFGS
from mace.calculators import mace_mp


water = smiles2conformers("O", numConfs=2)
print(water[0].info['connectivity'])
>>> [(0, 1, 1.0), (0, 2, 1.0)] # (atom_idx1, atom_idx2, bond_order)
ethanol = smiles2conformers("CCO", numConfs=5)
density = 1000  # kg/m^3
box = pack([water, ethanol], [7, 5], density)
print(box)
>>> Atoms(symbols='C10H44O12', pbc=True, cell=[8.4, 8.4, 8.4])
box.calc = mace_mp()  # MLIP calculator
opt = LBFGS(box)
opt.run(fmax=0.01)
```
All ASE Atoms objects generated or processed by `molify` store `connectivity` information (bonds and their orders) within the `ase.Atoms.info` dictionary.
If this information is available, `molify` uses it to convert between ASE, NetworkX and RDKit.
If an ASE structure is converted to an RDKit molecule without pre-existing connectivity, `molify` leverages RDKit's bond perception algorithms [@kimUniversalStructureConversion2015] to estimate this information.

A visualisation of the 2D and 3D structure from the simulation is shown in \autoref{fig:zndraw-rdkit}.

```python
from molify import ase2rdkit
from rdkit.Chem import Draw

mol = ase2rdkit(box)
img = Draw.MolToImage(mol)
```

Additionally, `molify` can convert structures to and from NetworkX graphs, enabling graph-based algorithms such as traversal, isomorphism checks, or component analysis:

```python
from molify import ase2networkx

graph = ase2networkx(box)  # NetworkX graph with atomic numbers and bond orders
```

This bidirectional conversion capability allows the use of RDKit's chemical analysis tools together with ASE for MLIP-based simulations.

For example, after atomic positions in an ASE Atoms object are updated, `molify` can convert this structure back to an RDKit molecule allowing for further analysis or visualization.
One common example is the extraction of substructures based on SMILES or SMARTS to track their structure and dynamics within a simulation.
For example, `molify` streamlines the extraction of the CH$_3$ alkyl group from the ethanol molecules inside the simulation cell, without manual index lookup.

```python
from molify import get_substructures

frames: list[ase.Atoms] = get_substructures(
  atoms=box,
  smiles="[C]([H])([H])[H]"
)
```

# Related software
The functionality of `molify` relies critically on the following packages:

- [RDKit](https://www.rdkit.org/docs/index.html): For cheminformatics tasks, SMILES parsing, conformer generation, and substructure searching.
- [ASE](https://wiki.fysik.dtu.dk/ase/): For representing and manipulating atomic structures, and interfacing with simulation engines.
- [PACKMOL](https://m3g.github.io/packmol/): For optimizing the placement of molecules within a specified simulation box size.
- [NetworkX](https://networkx.org/): For the handling and analysis of molecular graphs.

The `molify` package is currently a crucial part of the following software packages:

- [IPSuite](https://github.com/zincware/ipsuite): For generating structures for training MLIPs.
- [ZnDraw](https://github.com/zincware/zndraw): Interactive generation of simulation boxes and selection of substructures through a graphical user interface inside a web-based visualization package.
- [mlipx](https://github.com/basf/mlipx): Creating initial structures for benchmarking different MLIPs on real-world test scenarios.

The OpenBabel [@oboyleOpenBabelOpen2011] package provides similar cheminformatics functionality to RDKit along with extensive file format support.
However, OpenBabel is primarily designed as a format conversion tool with a focus on command-line usage and file I/O, while `molify` is designed for Python-native workflows with in-memory object conversions.
Currently, OpenBabel does not provide direct support for ASE Atoms objects, ASE calculators (including MLIPs), or seamless RDKit-ASE interconversion within Python.
Furthermore, `molify`'s integration with PACKMOL and NetworkX provides capabilities beyond OpenBabel's core focus on format conversion.

# Acknowledgements
F.Z. acknowledges support by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) in the framework of the priority program SPP 2363, "Utilization and Development of Machine Learning for Molecular Applications – Molecular Machine Learning" Project No. 497249646. Further funding through the DFG under Germany's Excellence Strategy – EXC 2075 – 390740016 and the Stuttgart Center for Simulation Science (SimTech) was provided.

# References
