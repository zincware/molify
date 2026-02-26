from typing import Literal

import ase
import networkx as nx
import numpy as np
from ase.neighborlist import natural_cutoffs, neighbor_list
from rdkit import Chem

try:
    import vesin
except ImportError:
    vesin = None

try:
    import xyzgraph as _xyzgraph
except ImportError:
    _xyzgraph = None


def _create_graph_from_connectivity(
    atoms: ase.Atoms, connectivity, charges
) -> nx.Graph:
    """Create NetworkX graph from explicit connectivity information."""
    graph = nx.Graph()
    graph.graph["pbc"] = atoms.pbc
    graph.graph["cell"] = atoms.cell

    for i, atom in enumerate(atoms):
        graph.add_node(
            i,
            position=atom.position,
            atomic_number=int(atom.number),
            original_index=atom.index,
            charge=charges[i],
        )

    for i, j, bond_order in connectivity:
        graph.add_edge(i, j, bond_order=bond_order)
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
        graph.nodes[i]["position"] = atom.position
        graph.nodes[i]["atomic_number"] = int(atom.number)
        graph.nodes[i]["original_index"] = atom.index
        graph.nodes[i]["charge"] = float(charges[i])
        if atom.number in non_bonding_atomic_numbers:
            graph.nodes[i]["charge"] = 1.0


def _xyzgraph_to_molify_graph(
    xg_graph: nx.Graph, atoms: ase.Atoms
) -> nx.Graph:
    """Convert an xyzgraph-produced NetworkX graph to molify's schema."""
    from ase.data import atomic_numbers

    graph = nx.Graph()
    graph.graph["pbc"] = atoms.pbc
    graph.graph["cell"] = atoms.cell

    for node_id, data in xg_graph.nodes(data=True):
        graph.add_node(
            node_id,
            atomic_number=atomic_numbers[data["symbol"]],
            position=np.array(data["position"]),
            original_index=node_id,
            charge=float(data.get("formal_charge", 0)),
        )

    for u, v, data in xg_graph.edges(data=True):
        graph.add_edge(u, v, bond_order=data["bond_order"])

    return graph


def _ase2networkx_xyzgraph(
    atoms: ase.Atoms,
    charge: int | None = None,
    **engine_kwargs,
) -> nx.Graph:
    """Build molecular graph using xyzgraph's cheminformatics pipeline."""
    from ase.data import chemical_symbols
    from molify.utils import unwrap_structures

    unwrapped = unwrap_structures(atoms, engine="rdkit")

    xyzgraph_atoms = [
        (chemical_symbols[atom.number], tuple(atom.position))
        for atom in unwrapped
    ]

    if charge is None:
        charge = int(sum(unwrapped.get_initial_charges()))

    xg_graph = _xyzgraph.build_graph(xyzgraph_atoms, charge=charge, **engine_kwargs)

    return _xyzgraph_to_molify_graph(xg_graph, atoms)


def ase2networkx(
    atoms: ase.Atoms,
    pbc: bool = True,
    scale: float = 1.2,
    engine: Literal["auto", "rdkit", "xyzgraph"] = "auto",
    charge: int | None = None,
    **engine_kwargs,
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
    engine : str, optional
        Backend engine for bond determination. One of ``"auto"``,
        ``"rdkit"``, or ``"xyzgraph"`` (default is ``"auto"``).
        ``"auto"`` uses xyzgraph if installed, otherwise falls back
        to the distance-based/rdkit pipeline.
    charge : int or None, optional
        Total molecular charge forwarded to xyzgraph (default is None).
    **engine_kwargs
        Additional keyword arguments forwarded to the engine backend.

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

    # Use explicit connectivity when present (regardless of engine)
    if "connectivity" in atoms.info:
        connectivity = atoms.info["connectivity"]
        # ensure connectivity is list[tuple[int, int, float|None]] and
        # does not contain np.generic
        connectivity = [
            (int(i), int(j), float(bond_order) if bond_order is not None else None)
            for i, j, bond_order in connectivity
        ]
        return _create_graph_from_connectivity(atoms, connectivity, charges)

    # Resolve engine (only reached when no explicit connectivity)
    use_xyzgraph = False
    if engine == "xyzgraph":
        if _xyzgraph is None:
            raise ImportError(
                "xyzgraph is required for engine='xyzgraph'. "
                "Install it with: pip install molify[xyzgraph]"
            )
        use_xyzgraph = True
    # engine == "auto" or "rdkit" -> use_xyzgraph stays False

    if use_xyzgraph:
        return _ase2networkx_xyzgraph(atoms, charge=charge, **engine_kwargs)

    connectivity_matrix, non_bonding_atomic_numbers = _compute_connectivity_matrix(
        atoms, scale, pbc
    )

    graph = nx.from_numpy_array(connectivity_matrix, edge_attr=None)
    for u, v in graph.edges():
        graph.edges[u, v]["bond_order"] = None

    _add_node_properties(graph, atoms, charges, non_bonding_atomic_numbers)

    graph.graph["pbc"] = atoms.pbc
    graph.graph["cell"] = atoms.cell

    return graph


def ase2rdkit(
    atoms: ase.Atoms,
    suggestions: list[str] | None = None,
    engine: Literal["auto", "rdkit", "xyzgraph"] = "auto",
    charge: int | None = None,
    **engine_kwargs,
) -> Chem.Mol:
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
    engine : Literal["auto", "rdkit", "xyzgraph"], optional
        Backend for bond detection and bond order assignment (default "auto").
        Passed through to ase2networkx().
    charge : int or None, optional
        Total system charge, forwarded to xyzgraph (default is None).
    **engine_kwargs
        Additional keyword arguments forwarded to the engine backend.

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

    graph = ase2networkx(atoms, engine=engine, charge=charge, **engine_kwargs)
    return networkx2rdkit(graph, suggestions=suggestions)
