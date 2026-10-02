import ase
import networkx as nx
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from rdkit import Chem

from molify.constants import EdgeAttr, GraphAttr, NodeAttr
from molify.utils import read_connectivity

try:
    import vesin
except ImportError:
    vesin = None

# Li, Na, K, Rb, Cs, Fr
_NON_BONDING_ATOMIC_NUMBERS = (3, 11, 19, 37, 55, 87)


def _create_graph_from_connectivity(
    atoms: ase.Atoms, connectivity, charges
) -> nx.Graph:
    """Create NetworkX graph from explicit connectivity information."""
    graph = nx.Graph()
    graph.graph[GraphAttr.PBC] = atoms.pbc
    graph.graph[GraphAttr.CELL] = atoms.cell

    stored_indices = atoms.info.get(NodeAttr.ORIGINAL_INDEX)

    for i, atom in enumerate(atoms):
        original_index = stored_indices[i] if stored_indices is not None else atom.index
        graph.add_node(
            i,
            **{
                NodeAttr.POSITION: atom.position,
                NodeAttr.ATOMIC_NUMBER: int(atom.number),
                NodeAttr.ORIGINAL_INDEX: original_index,
                NodeAttr.CHARGE: charges[i],
            },
        )

    for i, j, bond_order in connectivity:
        graph.add_edge(i, j, **{EdgeAttr.BOND_ORDER: bond_order})
    return graph


def _compute_bonded_pairs(atoms: ase.Atoms, scale: float, pbc: bool) -> np.ndarray:
    """Compute bonded atom pairs from distance-based cutoffs.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to search for bonds.
    scale : float
        Factor applied to the covalent radii.
    pbc : bool
        Whether bonds may cross periodic boundaries.

    Returns
    -------
    numpy.ndarray
        Shape ``(M, 2)``; unique ``(i, j)`` rows with ``i < j``, sorted
        lexicographically.
    """
    radii = covalent_radii[atoms.numbers] * scale
    bonding = ~np.isin(atoms.numbers, _NON_BONDING_ATOMIC_NUMBERS)
    if not bonding.any():
        return np.empty((0, 2), dtype=np.intp)
    # Neighbor lists keep d < cutoff; one ulp more keeps pairs at exactly the cutoff.
    max_cutoff = np.nextafter(2 * radii[bonding].max(), np.inf)

    if vesin is not None:
        try:
            i, j, d, s = vesin.ase_neighbor_list(
                "ijdS", atoms, cutoff=max_cutoff, self_interaction=False
            )
        except Exception as e:
            print(f"vesin failed with {e}, trying native ASE implementation")
            i, j, d, s = neighbor_list(
                "ijdS", atoms, cutoff=max_cutoff, self_interaction=False
            )
    else:
        i, j, d, s = neighbor_list(
            "ijdS", atoms, cutoff=max_cutoff, self_interaction=False
        )

    keep = (i < j) & bonding[i] & bonding[j] & (d <= radii[i] + radii[j])
    if not pbc:
        keep &= ~s.any(axis=1)
    return np.unique(np.stack([i[keep], j[keep]], axis=1), axis=0)


def _add_node_properties(graph: nx.Graph, atoms: ase.Atoms, charges):
    """Add node properties to the graph."""
    stored_indices = atoms.info.get(NodeAttr.ORIGINAL_INDEX)

    for i, atom in enumerate(atoms):
        original_index = stored_indices[i] if stored_indices is not None else atom.index
        graph.nodes[i][NodeAttr.POSITION] = atom.position
        graph.nodes[i][NodeAttr.ATOMIC_NUMBER] = int(atom.number)
        graph.nodes[i][NodeAttr.ORIGINAL_INDEX] = original_index
        graph.nodes[i][NodeAttr.CHARGE] = float(charges[i])
        if atom.number in _NON_BONDING_ATOMIC_NUMBERS:
            graph.nodes[i][NodeAttr.CHARGE] = 1.0


def ase2networkx(
    atoms: ase.Atoms,
    pbc: bool = True,
    scale: float = 1.2,
) -> nx.Graph:
    """Convert an ASE Atoms object to a NetworkX graph.

    Determines which atoms are bonded (connectivity).
    All edges will have bond_order=None unless atoms.info['connectivity']
    already has bond orders.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to convert into a graph.
    pbc : bool, optional
        Whether to consider periodic boundary conditions when calculating
        distances (default is True). If False, only connections within
        the unit cell are considered.
    scale : float, optional
        Scaling factor for the covalent radii when determining bond cutoffs
        (default is 1.2).

    Returns
    -------
    networkx.Graph
        An undirected NetworkX graph with connectivity information.

    Raises
    ------
    ValueError
        For an invalid bond in ``atoms.info['connectivity']``, see
        :func:`molify.utils.read_connectivity`.

    Notes
    -----
    The graph contains the following information:

    - Nodes represent atoms with properties:
        * position: Cartesian coordinates (numpy.ndarray)
        * atomic_number: Element atomic number (int)
        * original_index: Index in original Atoms object (int)
        * charge: Formal charge (float)
    - Edges represent bonds with:
        * bond_order: Bond order (float or None if unknown)
    - Graph properties include:
        * pbc: Periodic boundary conditions
        * cell: Unit cell vectors

    Connectivity is determined by:

    1. Using explicit connectivity if present in atoms.info
    2. Otherwise using distance-based cutoffs: atoms *i* and *j* are bonded
       when any periodic image of *j* lies within ``scale * (r_i + r_j)`` of
       *i*, with *r* the covalent radius. Li, Na, K, Rb, Cs and Fr are
       non-bonding ions. Edges have ``bond_order=None``; memory scales
       linearly with the number of atoms.

    To get bond orders, pass the graph to networkx2rdkit().

    Examples
    --------
    >>> from molify import ase2networkx, smiles2atoms
    >>> atoms = smiles2atoms(smiles="O")
    >>> graph = ase2networkx(atoms)
    >>> len(graph.nodes)
    3
    >>> len(graph.edges)
    2
    """
    if len(atoms) == 0:
        return nx.Graph()
    charges = atoms.get_initial_charges()

    if GraphAttr.CONNECTIVITY in atoms.info:
        connectivity = read_connectivity(atoms)
        return _create_graph_from_connectivity(atoms, connectivity, charges)

    pairs = _compute_bonded_pairs(atoms, scale, pbc)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    graph.add_edges_from(pairs.tolist(), **{EdgeAttr.BOND_ORDER: None})

    _add_node_properties(graph, atoms, charges)

    graph.graph[GraphAttr.PBC] = atoms.pbc
    graph.graph[GraphAttr.CELL] = atoms.cell

    return graph


def ase2rdkit(atoms: ase.Atoms, suggestions: list[str] | None = None) -> Chem.Mol:
    """Convert an ASE Atoms object to an RDKit molecule.

    Convenience function that chains:
    ase2networkx() → networkx2rdkit(suggestions=...)

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to convert.
    suggestions : list[str], optional
        SMILES/SMARTS patterns for bond order determination.
        Passed directly to networkx2rdkit().

    Returns
    -------
    rdkit.Chem.Mol
        The resulting RDKit molecule with bond orders determined.

    Examples
    --------
    >>> from molify import ase2rdkit, smiles2atoms
    >>> atoms = smiles2atoms(smiles="C=O")
    >>> mol = ase2rdkit(atoms)
    >>> mol.GetNumAtoms()
    4
    """
    if len(atoms) == 0:
        return Chem.Mol()

    from molify import ase2networkx, networkx2rdkit

    graph = ase2networkx(atoms)
    return networkx2rdkit(graph, suggestions=suggestions)
