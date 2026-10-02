import io
import numbers
from collections.abc import Sized
from typing import Literal, Optional, cast

import ase.io
import ase.units
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rdkit.Chem
import rdkit.Chem.rdDetermineBonds
from ase.data.colors import jmol_colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rdkit import Chem
from typing_extensions import deprecated

from molify.constants import GraphAttr


def _is_atom_index(value) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and float(value).is_integer()
    )


def read_connectivity(atoms: ase.Atoms) -> list[tuple[int, int, float | None]]:
    """Read ``atoms.info['connectivity']`` as native Python bonds.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure whose ``info['connectivity']`` holds ``(i, j, order)`` bonds,
        e.g. a list of tuples or the float array an extxyz file loads.

    Returns
    -------
    list[tuple[int, int, float | None]]
        One tuple per bond, in input order.

    Raises
    ------
    ValueError
        For an invalid bond. A valid bond has three entries ``(i, j, order)``
        with two distinct integer atom indices in ``0..len(atoms) - 1``.
    """
    n_atoms = len(atoms)
    bonds = []
    for bond in atoms.info[GraphAttr.CONNECTIVITY]:
        if not isinstance(bond, Sized) or len(bond) != 3:
            raise ValueError(
                f"bond {bond!r} in atoms.info['connectivity'] needs three entries "
                "(i, j, order)"
            )
        i, j, order = bond
        if not (_is_atom_index(i) and _is_atom_index(j)):
            raise ValueError(
                f"bond ({i!r}, {j!r}) in atoms.info['connectivity'] needs integer "
                "atom indices"
            )
        i, j = int(i), int(j)
        if not (0 <= i < n_atoms and 0 <= j < n_atoms):
            raise ValueError(
                f"bond ({i}, {j}) in atoms.info['connectivity'] needs atom "
                f"indices in the range 0..{n_atoms - 1}"
            )
        if i == j:
            raise ValueError(
                f"bond ({i}, {j}) in atoms.info['connectivity'] links atom {i} to "
                "itself"
            )
        bonds.append((i, j, None if order is None else float(order)))
    return bonds


def bond_type_from_order(order):
    if order == 1.0:
        return Chem.BondType.SINGLE
    elif order == 2.0:
        return Chem.BondType.DOUBLE
    elif order == 3.0:
        return Chem.BondType.TRIPLE
    elif order == 1.5:
        return Chem.BondType.AROMATIC
    else:
        raise ValueError(f"Unsupported bond order: {order}")


def fragment_indices(atoms: ase.Atoms) -> list[list[int]]:
    """Group atom indices into the molecules bonded by ``info['connectivity']``.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure with ``info['connectivity']``.

    Returns
    -------
    list[list[int]]
        Sorted atom indices per molecule, ordered by first atom. An unbonded
        atom is a molecule of its own.

    Raises
    ------
    ValueError
        For an invalid bond in ``atoms.info['connectivity']``, see
        :func:`read_connectivity`.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    graph.add_edges_from((i, j) for i, j, _ in read_connectivity(atoms))
    return sorted(sorted(component) for component in nx.connected_components(graph))


@deprecated(
    "find_connected_components is deprecated; use molify.utils.fragment_indices(atoms)"
)
def find_connected_components(connectivity: list[tuple[int, int, float]]):
    graph = nx.Graph()
    graph.add_edges_from((i, j) for i, j, _ in connectivity)
    yield from nx.connected_components(graph)


def calculate_density(atoms: ase.Atoms) -> float:
    """Calculates the density of an ASE Atoms object in kg/m^3.

    The density is calculated as the total mass (in kg) divided by the volume (in m^3).
    """
    total_mass_kg = sum(atoms.get_masses()) * ase.units._amu  # amu -> kg
    volume_m3 = atoms.get_volume() * 1e-30  # Å^3 -> m^3
    density = total_mass_kg / volume_m3  # kg/m^3
    return density


def calculate_box_dimensions(images: list[ase.Atoms], density: float) -> list[float]:
    """Calculates the dimensions of the simulation box

    based on the molar volume and target density.
    """
    total_mass = sum(sum(atoms.get_masses()) for atoms in images)
    molar_volume = total_mass / density / 1000  # m^3 / mol
    volume_per_mol = molar_volume * ase.units.m**3 / ase.units.mol
    box_edge = volume_per_mol ** (1 / 3)
    return [box_edge] * 3


def unwrap_structures(atoms, scale=1.2, **kwargs) -> ase.Atoms:
    """Unwrap molecular structures across periodic boundary conditions (PBC).

    This function corrects atomic positions that have been wrapped across periodic
    boundaries, ensuring that bonded atoms appear as continuous molecular structures.
    It can handle multiple disconnected molecules within the same unit cell.

    The algorithm works by:
    1. Building a connectivity graph based on covalent radii
    2. Traversing each connected component (molecule) using depth-first search
    3. Accumulating periodic image shifts to maintain molecular connectivity
    4. Applying the shifts to obtain unwrapped coordinates

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object containing wrapped atomic positions
    scale : float, optional
        Scale factor for covalent radii cutoffs used in bond detection.
        Larger values include more distant neighbors as bonded.
        Default is 1.2.
    **kwargs : dict
        Additional keyword arguments to pass to the `ase2networkx` function.

    Returns
    -------
    ase.Atoms
        A new ASE Atoms object with unwrapped atomic positions. The original
        atoms object is not modified.

    Notes
    -----
    - The function preserves the original atoms object and returns a copy
    - Works with any periodic boundary conditions (1D, 2D, or 3D)
    - Handles multiple disconnected molecules/fragments
    - Uses covalent radii scaled by the `scale` parameter for bond detection

    Examples
    --------
    >>> import numpy as np
    >>> from molify import smiles2conformers, pack, unwrap_structures
    >>>
    >>> # Create a realistic molecular system
    >>> water = smiles2conformers("O", numConfs=1)
    >>> ethanol = smiles2conformers("CCO", numConfs=1)
    >>>
    >>> # Pack molecules into a periodic box
    >>> packed_system = pack(
    ...     data=[water, ethanol],
    ...     counts=[10, 5],
    ...     density=800
    ... )
    >>>
    >>> # Simulate wrapped coordinates (as might occur in MD)
    >>> cell = packed_system.get_cell()
    >>> positions = packed_system.get_positions()
    >>>
    >>> # Artificially move the box and wrap it again
    >>> positions += np.array([cell[0, 0] * 0.5, 0, 0])  # Shift by half box length
    >>> wrapped_atoms = packed_system.copy()
    >>> wrapped_atoms.set_positions(positions)
    >>> wrapped_atoms.wrap()  # Wrap back into PBC
    >>>
    >>> # Unwrap the structures to get continuous molecules
    >>> unwrapped_atoms = unwrap_structures(wrapped_atoms)
    """
    from molify import ase2networkx

    atoms = atoms.copy()  # Work on a copy to avoid modifying the original

    graph = ase2networkx(atoms, scale=scale, **kwargs)
    positions = atoms.get_positions()
    cell = atoms.get_cell()

    for component in nx.connected_components(graph):
        if len(component) == 1:
            continue
        # start at the atom closest to the center of the box
        root = min(
            component, key=lambda i: np.linalg.norm(positions[i] - cell.diagonal() / 2)
        )
        # now do a bfs traversal to unwrap the molecule
        for i, j in nx.dfs_tree(graph, source=root).edges():
            # i has already been unwrapped
            # j is the neighbor that needs to be unwrapped
            offsets = cell.scaled_positions(positions[i] - positions[j])
            offsets = offsets.round().astype(int)
            positions[j] += offsets @ cell  # Unwrap j to the same image as i
    atoms.set_positions(positions)
    # TODO: include connectivity information
    return atoms


def rdkit_determine_bonds(unwrapped_atoms: ase.Atoms) -> rdkit.Chem.Mol:
    if len(unwrapped_atoms) == 0:
        raise ValueError("Cannot determine bonds for an empty structure.")
    with io.StringIO() as f:
        ase.io.write(f, unwrapped_atoms, format="xyz")
        f.seek(0)
        xyz = f.read()
        mol = rdkit.Chem.MolFromXYZBlock(xyz)
    if len(unwrapped_atoms) == 1:
        if unwrapped_atoms.get_chemical_symbols()[0] in ["Li", "Na", "K", "Rb", "Cs"]:
            return rdkit.Chem.MolFromSmiles(
                f"[{unwrapped_atoms.get_chemical_symbols()[0]}+]"
            )
        if unwrapped_atoms.get_chemical_symbols()[0] in ["Cl", "Br", "I", "F"]:
            return rdkit.Chem.MolFromSmiles(
                f"[{unwrapped_atoms.get_chemical_symbols()[0]}-]"
            )
    if len(unwrapped_atoms) == 7:
        if sorted(unwrapped_atoms.get_chemical_symbols()) == sorted(
            ["P", "F", "F", "F", "F", "F", "F"]
        ):
            return rdkit.Chem.MolFromSmiles("F[P-](F)(F)(F)(F)F")
    for charge in [0, 1, -1, 2, -2]:
        try:
            rdkit.Chem.rdDetermineBonds.DetermineBonds(
                mol,
                charge=int(sum(unwrapped_atoms.get_initial_charges())) + charge,
            )
            return mol
        except ValueError:
            pass
    else:
        raise ValueError(
            "Failed to determine bonds for sub-structure up to charge "
            f"{sum(unwrapped_atoms.get_initial_charges()) + charge} "
            f"and {unwrapped_atoms.get_chemical_symbols()}"
        )


def suggestions2networkx(smiles: list[str]) -> list[nx.Graph]:
    from molify import rdkit2networkx

    mols = []
    for _smiles in smiles:
        mol = Chem.MolFromSmiles(_smiles)
        mol = Chem.AddHs(mol)
        mols.append(mol)
    return [rdkit2networkx(mol) for mol in mols]


def draw_molecular_graph(  # noqa: C901
    graph: nx.Graph,
    layout: Literal["spring", "circular", "kamada_kawai"] = "kamada_kawai",
    figsize: tuple[float, float] = (8, 8),
    node_size: int = 500,
    font_size: Optional[int] = None,
    show_bond_orders: bool = True,
    weight_by_bond_order: bool = True,
    ax: Optional[Axes] = None,
) -> Figure:
    """Draw a molecular graph with bond order visualization.

    This function visualizes NetworkX molecular graphs with proper representation
    of chemical bonds. Single, double, triple, and aromatic bonds are displayed
    with appropriate line styles.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph with node attribute 'atomic_number' and edge attribute
        'bond_order'. Can be created from molify conversion functions like
        ase2networkx() or rdkit2networkx().
    layout : Literal["spring", "circular", "kamada_kawai"], optional
        Layout algorithm to use for node positioning. Options are:
            * `spring`: Force-directed layout
            * `circular`: Nodes arranged in a circle
            * `kamada_kawai`: Force-directed using Kamada-Kawai algorithm (default)
    figsize : tuple[float, float], optional
        Figure size as (width, height) in inches. Default is (8, 8).
    node_size : int, optional
        Size of the nodes. Default is 500.
    font_size : int, optional
        Font size for node labels (atomic numbers). If None, automatically scaled
        based on node_size (approximately 0.66 * sqrt(node_size)). Default is None.
    show_bond_orders : bool, optional
        If True, visualize different bond orders with multiple parallel lines.
        Default is True.
    weight_by_bond_order : bool, optional
        If True and layout is 'spring', use bond_order as edge weights in the
        spring layout algorithm. Higher bond orders pull atoms closer together.
        This makes double/triple bonds appear shorter than single bonds.
        Default is True.
    ax : Axes, optional
        Matplotlib axes to draw on. If None, creates a new figure.

    Returns
    -------
    Figure
        Matplotlib figure containing the molecular graph visualization.

    Notes
    -----
    **Node Colors:**

    - Nodes are colored using Jmol color scheme based on atomic numbers
    - Common colors: Carbon (gray), Hydrogen (white), Oxygen (red),
      Nitrogen (blue)

    **Bond Order Visualization:**

    - Single bond (1.0): 1 solid line
    - Double bond (2.0): 2 parallel solid lines
    - Triple bond (3.0): 3 parallel solid lines
    - Aromatic bond (1.5): 1 solid line + 1 dashed line (parallel)
    - Unknown (None): 1 thin solid line

    Examples
    --------
    >>> from molify import smiles2atoms, ase2networkx, draw_molecular_graph
    >>> atoms = smiles2atoms("C=C=O")  # Ketene
    >>> graph = ase2networkx(atoms)
    >>> fig = draw_molecular_graph(graph, layout='spring')

    >>> # For benzene showing aromatic bonds
    >>> benzene = smiles2atoms("c1ccccc1")
    >>> graph_benzene = ase2networkx(benzene)
    >>> fig = draw_molecular_graph(graph_benzene, show_bond_orders=True)
    """
    # Calculate font_size and linewidth based on node_size if not provided
    if font_size is None:
        font_size = int(0.66 * np.sqrt(node_size))

    linewidth = 0.091 * np.sqrt(node_size)

    # Create figure if axes not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig_or_none = ax.get_figure()
        if fig_or_none is None:
            raise ValueError("Provided axes is not attached to a figure")
        # Cast to Figure (assuming user provides proper Axes, not SubFigure axes)
        fig = cast(Figure, fig_or_none)
        created_fig = False

    # Compute node positions using the specified layout
    if layout == "spring":
        if weight_by_bond_order:
            # Use bond_order as weights - higher order = stronger spring
            pos = nx.spring_layout(graph, weight="bond_order", seed=42)
        else:
            pos = nx.spring_layout(graph, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        raise ValueError(
            f"Unknown layout '{layout}'. "
            f"Choose from 'spring', 'circular', 'kamada_kawai'."
        )

    # Draw edges with bond order visualization
    if show_bond_orders:
        _draw_edges_with_bond_orders(graph, pos, ax)
    else:
        nx.draw_networkx_edges(graph, pos, ax=ax, width=2)

    # Get node colors based on atomic numbers using Jmol colors
    node_colors = []
    for node in graph.nodes():
        atomic_number = graph.nodes[node].get("atomic_number", 0)
        # jmol_colors is indexed by atomic number (1-indexed for elements)
        if atomic_number > 0 and atomic_number < len(jmol_colors):
            node_colors.append(jmol_colors[atomic_number])
        else:
            node_colors.append([0.5, 0.5, 0.5])  # Gray for unknown

    # Draw nodes with Jmol colors
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_size,
        node_color=node_colors,
        edgecolors="black",
        linewidths=linewidth,
        ax=ax,
    )

    # Draw labels (atomic numbers)
    labels = nx.get_node_attributes(graph, "atomic_number")
    nx.draw_networkx_labels(
        graph, pos, labels=labels, font_size=font_size, font_color="black", ax=ax
    )

    # Add margins to prevent node clipping at edges
    ax.margins(0.15)
    ax.axis("off")
    fig.tight_layout()

    # Close figure to prevent double display in notebooks
    # (only if we created it; otherwise it's managed externally)
    if created_fig:
        plt.close(fig)

    return fig


def _draw_edges_with_bond_orders(
    graph: nx.Graph, pos: dict, ax: Axes, line_offset: float = 0.02
) -> None:
    """Draw edges with multiple parallel lines based on bond order.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph with edge attribute 'bond_order'
    pos : dict
        Dictionary mapping node IDs to (x, y) positions
    ax : Axes
        Matplotlib axes to draw on
    line_offset : float
        Perpendicular offset between parallel lines for multiple bonds
    """
    for u, v, data in graph.edges(data=True):
        bond_order = data.get("bond_order")

        # Get node positions
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        # Calculate perpendicular direction for parallel lines
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)

        if length == 0:
            continue

        # Perpendicular unit vector
        perp_x = -dy / length
        perp_y = dx / length

        if bond_order is None:
            # Unknown bond order - draw thin line
            ax.plot([x0, x1], [y0, y1], "k-", linewidth=1, zorder=1)

        elif bond_order == 1.0:
            # Single bond
            ax.plot([x0, x1], [y0, y1], "k-", linewidth=2, zorder=1)

        elif bond_order == 2.0:
            # Double bond - two parallel lines
            offset = line_offset
            for sign in [-1, 1]:
                x0_offset = x0 + sign * offset * perp_x
                y0_offset = y0 + sign * offset * perp_y
                x1_offset = x1 + sign * offset * perp_x
                y1_offset = y1 + sign * offset * perp_y
                ax.plot(
                    [x0_offset, x1_offset],
                    [y0_offset, y1_offset],
                    "k-",
                    linewidth=2,
                    zorder=1,
                )

        elif bond_order == 3.0:
            # Triple bond - three parallel lines
            offset = line_offset
            # Center line
            ax.plot([x0, x1], [y0, y1], "k-", linewidth=2, zorder=1)
            # Two outer lines
            for sign in [-1, 1]:
                x0_offset = x0 + sign * offset * perp_x
                y0_offset = y0 + sign * offset * perp_y
                x1_offset = x1 + sign * offset * perp_x
                y1_offset = y1 + sign * offset * perp_y
                ax.plot(
                    [x0_offset, x1_offset],
                    [y0_offset, y1_offset],
                    "k-",
                    linewidth=2,
                    zorder=1,
                )

        elif bond_order == 1.5:
            # Aromatic bond - one solid and one dashed line
            offset = line_offset * 0.7
            # Solid line
            x0_offset = x0 - offset * perp_x
            y0_offset = y0 - offset * perp_y
            x1_offset = x1 - offset * perp_x
            y1_offset = y1 - offset * perp_y
            ax.plot(
                [x0_offset, x1_offset],
                [y0_offset, y1_offset],
                "k-",
                linewidth=2,
                zorder=1,
            )
            # Dashed line
            x0_offset = x0 + offset * perp_x
            y0_offset = y0 + offset * perp_y
            x1_offset = x1 + offset * perp_x
            y1_offset = y1 + offset * perp_y
            ax.plot(
                [x0_offset, x1_offset],
                [y0_offset, y1_offset],
                "k--",
                linewidth=2,
                zorder=1,
            )

        else:
            # Unknown bond order value - draw single line
            ax.plot([x0, x1], [y0, y1], "k-", linewidth=2, zorder=1)
