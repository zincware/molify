import ase
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

from molify.constants import EdgeAttr, GraphAttr, NodeAttr


def rdkit2ase(mol: Chem.Mol, seed: int = 42) -> ase.Atoms:
    """Convert an RDKit molecule to an ASE Atoms object.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule to convert
    seed : int, optional
        Random seed for conformer generation (default is 42)

    Returns
    -------
    ase.Atoms
        ASE Atoms object with:
        - Atomic positions from conformer coordinates
        - Atomic numbers from molecular structure
        - Connectivity information in atoms.info
        - Formal charges if present
        - SMILES string in atoms.info

    Examples
    --------
    >>> from rdkit import Chem
    >>> from molify import rdkit2ase
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atoms = rdkit2ase(mol)
    >>> len(atoms)
    9
    """
    if mol.GetNumAtoms() == 0:
        return ase.Atoms()

    smiles = Chem.MolToSmiles(mol)
    mol = Chem.AddHs(mol)
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    atoms = ase.Atoms(
        positions=mol.GetConformer().GetPositions(),
        numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
    )
    atoms.info[GraphAttr.SMILES] = smiles
    atoms.info[GraphAttr.CONNECTIVITY] = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        atoms.info[GraphAttr.CONNECTIVITY].append((a1, a2, order))

    if any(charge != 0 for charge in charges):
        atoms.set_initial_charges(charges)

    if all(atom.HasProp(NodeAttr.ORIGINAL_INDEX) for atom in mol.GetAtoms()):
        atoms.info[NodeAttr.ORIGINAL_INDEX] = [
            atom.GetIntProp(NodeAttr.ORIGINAL_INDEX) for atom in mol.GetAtoms()
        ]

    return atoms


def rdkit2networkx(mol: Chem.Mol) -> nx.Graph:
    """Convert an RDKit molecule to a NetworkX graph.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object to be converted

    Returns
    -------
    networkx.Graph
        Undirected graph representing the molecule where:

        Nodes contain:
            * atomic_number (int): Atomic number
            * original_index (int): RDKit atom index
            * charge (int): Formal charge
        Edges contain:
            * bond_order (float): Bond order (1.0, 1.5, 2.0, or 3.0)

    Notes
    -----
    Bond orders are converted as follows:
        * SINGLE -> 1.0
        * DOUBLE -> 2.0
        * TRIPLE -> 3.0
        * AROMATIC -> 1.5

    Examples
    --------
    >>> from rdkit import Chem
    >>> from molify import rdkit2networkx
    >>> mol = Chem.MolFromSmiles('C=O')
    >>> graph = rdkit2networkx(mol)
    >>> len(graph.nodes)
    4
    >>> len(graph.edges)
    3
    """
    mol = Chem.AddHs(mol)
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        original_index = (
            atom.GetIntProp(NodeAttr.ORIGINAL_INDEX)
            if atom.HasProp(NodeAttr.ORIGINAL_INDEX)
            else atom.GetIdx()
        )
        graph.add_node(
            atom.GetIdx(),
            **{
                NodeAttr.ATOMIC_NUMBER: atom.GetAtomicNum(),
                NodeAttr.ORIGINAL_INDEX: original_index,
                NodeAttr.CHARGE: atom.GetFormalCharge(),
            },
        )

    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        bond_order = 0
        if bond_type == Chem.BondType.SINGLE:
            bond_order = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_order = 2
        elif bond_type == Chem.BondType.TRIPLE:
            bond_order = 3
        elif bond_type == Chem.BondType.AROMATIC:
            bond_order = 1.5

        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            **{EdgeAttr.BOND_ORDER: bond_order},
        )
    return graph
