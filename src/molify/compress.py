import ase
import numpy as np
from ase.cell import Cell

from molify.constants import GraphAttr
from molify.utils import calculate_box_dimensions


def compress(
    atoms: ase.Atoms,
    density: float,
    freeze_molecules: bool = False,
) -> ase.Atoms:
    """Compress an ASE Atoms object to a target density.

    Arguments
    ---------
    atoms : ase.Atoms
        The Atoms object to compress.
    density : float
        The target density in g/cm^3.
    freeze_molecules : bool
        If True, freeze the internal degrees of freedom of the molecules
        during compression, to prevent bond compression.
    """
    atoms = atoms.copy()
    new_dimensions = np.array(calculate_box_dimensions([atoms], density))

    if freeze_molecules:
        new_cell = Cell.new(new_dimensions)
        old_cell = atoms.get_cell()
        connectivity = atoms.info.get(GraphAttr.CONNECTIVITY)
        if connectivity is None:
            raise ValueError("No connectivity info found for freeze_molecules=True")

        # Build molecular fragments
        from collections import defaultdict

        groups = defaultdict(set)
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i, j, _ in connectivity:
            union(i, j)

        for idx in range(len(atoms)):
            root = find(idx)
            groups[root].add(idx)

        t_mat = np.linalg.solve(old_cell.T, new_cell.T)

        positions = atoms.get_positions()
        for group in groups.values():
            group = list(group)
            com = atoms[group].get_center_of_mass()
            com_new = com @ t_mat
            shift = com_new - com
            positions[group] += shift

        atoms.set_positions(positions)
        atoms.set_cell(new_cell, scale_atoms=False)
    else:
        atoms.set_cell(new_dimensions, scale_atoms=True)

    return atoms
