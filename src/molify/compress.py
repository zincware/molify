import ase
import numpy as np
from ase.cell import Cell

from molify.constants import GraphAttr
from molify.utils import calculate_box_dimensions, fragment_indices


def compress(
    atoms: ase.Atoms,
    density: float,
    freeze_molecules: bool = False,
) -> ase.Atoms:
    """Compress an ASE Atoms object to a target density.

    Parameters
    ----------
    atoms : ase.Atoms
        The Atoms object to compress.
    density : float
        The target density in kg/m^3.
    freeze_molecules : bool
        If True, freeze the internal degrees of freedom of the molecules
        during compression, to prevent bond compression.

    Raises
    ------
    ValueError
        With ``freeze_molecules=True``, for a missing ``info['connectivity']``
        or an invalid bond in it, see :func:`molify.utils.read_connectivity`.
    """
    atoms = atoms.copy()
    new_dimensions = np.array(calculate_box_dimensions([atoms], density))

    if freeze_molecules:
        if atoms.info.get(GraphAttr.CONNECTIVITY) is None:
            raise ValueError("No connectivity info found for freeze_molecules=True")
        fragments = fragment_indices(atoms)
        new_cell = Cell.new(new_dimensions)
        t_mat = np.linalg.solve(atoms.get_cell().T, new_cell.T)

        positions = atoms.get_positions()
        for fragment in fragments:
            com = atoms[fragment].get_center_of_mass()
            positions[fragment] += com @ t_mat - com

        atoms.set_positions(positions)
        atoms.set_cell(new_cell, scale_atoms=False)
    else:
        atoms.set_cell(new_dimensions, scale_atoms=True)

    return atoms
