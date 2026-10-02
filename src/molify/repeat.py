import numbers
from collections.abc import Sequence

import ase
import numpy as np
from ase.geometry import find_mic

from molify.constants import GraphAttr, NodeAttr


def _is_positive_int(value: object) -> bool:
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, bool)
        and value >= 1
    )


def _normalize_rep(rep: int | Sequence[int]) -> tuple[int, int, int]:
    error = ValueError(
        "rep must be a positive integer or a sequence of three positive integers, "
        f"got {rep!r}"
    )
    if isinstance(rep, numbers.Integral):
        values = (rep,) * 3
    elif isinstance(rep, str):
        raise error
    else:
        try:
            values = tuple(rep)
        except TypeError:
            raise error from None
    if len(values) != 3 or not all(_is_positive_int(value) for value in values):
        raise error
    return tuple(int(value) for value in values)


def _read_bonds(atoms: ase.Atoms) -> tuple[np.ndarray, np.ndarray, list]:
    n_atoms = len(atoms)
    first, second, orders = [], [], []
    for bond in atoms.info[GraphAttr.CONNECTIVITY]:
        i, j = int(bond[0]), int(bond[1])
        if not (0 <= i < n_atoms and 0 <= j < n_atoms):
            raise ValueError(
                f"bond ({i}, {j}) in atoms.info['connectivity'] needs atom "
                f"indices in the range 0..{n_atoms - 1}"
            )
        first.append(i)
        second.append(j)
        orders.append(bond[2])
    return np.array(first, dtype=int), np.array(second, dtype=int), orders


def _tile_bonds(
    atoms: ase.Atoms,
    i: np.ndarray,
    j: np.ndarray,
    orders: list,
    reps: tuple[int, int, int],
) -> list[tuple[int, int, float | None]]:
    if len(orders) == 0:
        return []
    d = atoms.positions[j] - atoms.positions[i]
    d_mic, _ = find_mic(d, atoms.cell, atoms.pbc)
    shift = np.rint(atoms.cell.scaled_positions(d_mic - d)).astype(int)

    n_atoms = len(atoms)
    n_images = int(np.prod(reps))
    images = np.array(list(np.ndindex(*reps)))
    targets = (images[:, None, :] + shift[None, :, :]) % np.array(reps)
    target_image = np.ravel_multi_index(np.moveaxis(targets, -1, 0), reps)
    new_i = np.arange(n_images)[:, None] * n_atoms + i[None, :]
    new_j = target_image * n_atoms + j[None, :]
    return [
        (int(a), int(b), order)
        for a, b, order in zip(
            new_i.ravel(), new_j.ravel(), orders * n_images, strict=True
        )
    ]


def repeat(atoms: ase.Atoms, rep: int | Sequence[int]) -> ase.Atoms:
    """Repeat a periodic structure together with its connectivity.

    Atom ``k`` of the result is a copy of atom ``k % len(atoms)``, as in
    :meth:`ase.Atoms.repeat`. Each bond ``(i, j, order)`` links atom ``i`` to
    the nearest periodic image of atom ``j``, so bonds across the cell
    boundary connect neighbouring copies. Bond orders are kept.

    Parameters
    ----------
    atoms : ase.Atoms
        Periodic structure.
    rep : int or Sequence[int]
        Copies along each cell vector: one positive integer for all three, or
        three positive integers.

    Returns
    -------
    ase.Atoms
        Repeated structure. ``info['connectivity']`` and
        ``info['original_index']`` are tiled when present; other ``info``
        entries follow :meth:`ase.Atoms.repeat`.

    Raises
    ------
    ValueError
        For an invalid ``rep``, an ``info['original_index']`` of the wrong
        length, a bond index outside ``0..len(atoms) - 1``, or a repeat along
        an undefined cell vector.

    Examples
    --------
    >>> import molify
    >>> water = molify.smiles2conformers("O", numConfs=1)
    >>> box = molify.pack([water], counts=[4], density=1000)
    >>> big = molify.repeat(box, (2, 1, 1))
    >>> len(big.info["connectivity"]) == 2 * len(box.info["connectivity"])
    True
    """
    reps = _normalize_rep(rep)
    original_index = atoms.info.get(NodeAttr.ORIGINAL_INDEX)
    if original_index is not None and len(original_index) != len(atoms):
        raise ValueError(
            f"atoms.info['original_index'] holds {len(original_index)} entries "
            f"for {len(atoms)} atoms; it needs one entry per atom"
        )
    bonds = _read_bonds(atoms) if GraphAttr.CONNECTIVITY in atoms.info else None

    result = atoms.repeat(reps)
    if bonds is not None:
        result.info[GraphAttr.CONNECTIVITY] = _tile_bonds(atoms, *bonds, reps)
    if original_index is not None:
        tiled = [int(k) for k in original_index] * int(np.prod(reps))
        result.info[NodeAttr.ORIGINAL_INDEX] = tiled
    return result
