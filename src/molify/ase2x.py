import ase
import networkx as nx
import numpy as np
from ase.neighborlist import natural_cutoffs, neighbor_list
from rdkit import Chem

from molify.constants import EdgeAttr, GraphAttr, NodeAttr

try:
    import vesin
except ImportError:
    vesin = None


def _create_graph_from_connectivity(
    atoms: ase.Atoms, connectivity, charges
) -> nx.Graph:
    """Create NetworkX graph from explicit connectivity information."""
    graph = nx.Graph()
    graph.graph[GraphAttr.PBC] = atoms.pbc
    graph.graph[GraphAttr.CELL] = atoms.cell

    for i, atom in enumerate(atoms):
        graph.add_node(
            i,
            **{
                NodeAttr.POSITION: atom.position,
                NodeAttr.ATOMIC_NUMBER: int(atom.number),
                NodeAttr.ORIGINAL_INDEX: atom.index,
                NodeAttr.CHARGE: charges[i],
            },
        )

    for i, j, bond_order in connectivity:
        graph.add_edge(i, j, **{EdgeAttr.BOND_ORDER: bond_order})
    return graph


def _compute_connectivity_matrix(atoms: ase.Atoms, scale: float, pbc: bool):
    """Compute connectivity matrix from distance-based cutoffs."""
    # non-bonding positive charged atoms / ions.
    non_bonding_atomic_numbers = {3, 11, 19, 37, 55, 87}

    atomic_numbers = atoms.get_atomic_numbers()
    excluded_mask = np.isin(atomic_numbers, list(non_bonding_atomic_numbers))

    atom_radii = np.array(natural_cutoffs(atoms, mult=scale))
    pairwise_cutoffs = atom_radii[:, None] + atom_radii[None, :]
    max_cutoff = np.max(pairwise_cutoffs)

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

    # If pbc=False, filter out bonds that cross periodic boundaries
    if not pbc:
        non_periodic_mask = np.all(s == 0, axis=1)
        i = i[non_periodic_mask]
        j = j[non_periodic_mask]
        d = d[non_periodic_mask]

    d_ij = np.full((len(atoms), len(atoms)), np.inf)
    d_ij[i, j] = d
    np.fill_diagonal(d_ij, 0.0)

    # mask out non-bonding atoms
    d_ij[excluded_mask, :] = np.inf
    d_ij[:, excluded_mask] = np.inf

    connectivity_matrix = np.zeros((len(atoms), len(atoms)), dtype=int)
    np.fill_diagonal(d_ij, np.inf)
    connectivity_matrix[d_ij <= pairwise_cutoffs] = 1

    return connectivity_matrix, non_bonding_atomic_numbers


def _add_node_properties(
    graph: nx.Graph, atoms: ase.Atoms, charges, non_bonding_atomic_numbers
):
    """Add node properties to the graph."""
    for i, atom in enumerate(atoms):
        graph.nodes[i][NodeAttr.POSITION] = atom.position
        graph.nodes[i][NodeAttr.ATOMIC_NUMBER] = int(atom.number)
        graph.nodes[i][NodeAttr.ORIGINAL_INDEX] = atom.index
        graph.nodes[i][NodeAttr.CHARGE] = float(charges[i])
        if atom.number in non_bonding_atomic_numbers:
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
    2. Otherwise using distance-based cutoffs (edges will have bond_order=None)

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
        connectivity = atoms.info[GraphAttr.CONNECTIVITY]
        # ensure connectivity is list[tuple[int, int, float|None]] and
        # does not contain np.generic
        connectivity = [
            (int(i), int(j), float(bond_order) if bond_order is not None else None)
            for i, j, bond_order in connectivity
        ]
        return _create_graph_from_connectivity(atoms, connectivity, charges)

    connectivity_matrix, non_bonding_atomic_numbers = _compute_connectivity_matrix(
        atoms, scale, pbc
    )

    graph = nx.from_numpy_array(connectivity_matrix, edge_attr=None)
    for u, v in graph.edges():
        graph.edges[u, v][EdgeAttr.BOND_ORDER] = None

    _add_node_properties(graph, atoms, charges, non_bonding_atomic_numbers)

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
