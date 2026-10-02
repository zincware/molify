from collections.abc import Sequence

import ase
import numpy as np
from ase.geometry import find_mic, minkowski_reduce

from molify.constants import GraphAttr, NodeAttr


def _normalize_rep(rep: int | Sequence[int]) -> tuple[int, ...]:
    try:
        reps = np.asarray(rep)
    except ValueError:
        reps = np.empty(0)
    if reps.ndim == 0:
        reps = np.broadcast_to(reps, 3)
    if reps.shape != (3,) or reps.dtype.kind not in "iu" or (reps < 1).any():
        raise ValueError(
            "rep must be a positive integer or a sequence of three positive integers, "
            f"got {rep!r}"
        )
    return tuple(reps.tolist())


def _tile_bonds(
    atoms: ase.Atoms, reps: tuple[int, ...]
) -> list[tuple[int, int, float | None]]:
    n_atoms = len(atoms)
    first, second, orders = [], [], []
    for bond in atoms.info[GraphAttr.CONNECTIVITY]:
        i, j = int(bond[0]), int(bond[1])
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
        first.append(i)
        second.append(j)
        orders.append(None if bond[2] is None else float(bond[2]))
    i = np.array(first, dtype=int)
    j = np.array(second, dtype=int)

    periodic = atoms.pbc & atoms.cell.any(1)
    d = atoms.positions[j] - atoms.positions[i]
    d_mic, lengths = find_mic(d, atoms.cell, periodic)
    lattice, _ = minkowski_reduce(atoms.cell, periodic)
    limit = 0.5 * np.linalg.norm(lattice[periodic], axis=1).min(initial=np.inf)
    too_long = np.flatnonzero(lengths >= limit)
    if too_long.size:
        k = too_long[0]
        raise ValueError(
            f"bond ({i[k]}, {j[k]}) in atoms.info['connectivity'] is "
            f"{lengths[k]:.3f} Å long; repeat needs every bond shorter than "
            f"{limit:.3f} Å, half the shortest periodic lattice vector, so that "
            "each bond links a unique periodic image"
        )
    shift = np.rint(atoms.cell.scaled_positions(d_mic - d)).astype(int)

    images = np.array(list(np.ndindex(*reps)))
    targets = (images[:, None, :] + shift[None, :, :]) % np.array(reps)
    target_image = np.ravel_multi_index(np.moveaxis(targets, -1, 0), reps)
    new_i = np.arange(len(images))[:, None] * n_atoms + i[None, :]
    new_j = target_image * n_atoms + j[None, :]
    return [
        (int(a), int(b), order)
        for a, b, order in zip(
            new_i.ravel(), new_j.ravel(), orders * len(images), strict=True
        )
    ]


def repeat(atoms: ase.Atoms, rep: int | Sequence[int]) -> ase.Atoms:
    """Repeat a periodic structure together with its connectivity.

    Atom ``k`` of the result is a copy of atom ``k % len(atoms)``, as in
    :meth:`ase.Atoms.repeat`. Each bond ``(i, j, order)`` links atom ``i`` to
    the nearest periodic image of atom ``j``. Bond orders are kept.

    Parameters
    ----------
    atoms : ase.Atoms
        Periodic structure whose bonds are each shorter than half the shortest
        vector of its periodic lattice.
    rep : int or Sequence[int]
        Copies along each cell vector: one positive integer for all three, or
        three positive integers.

    Returns
    -------
    ase.Atoms
        Repeated structure with ``info['connectivity']`` and
        ``info['original_index']`` tiled and ``info['smiles']`` removed.

    Raises
    ------
    ValueError
        For an invalid ``rep``, an ``info['original_index']`` of the wrong
        length, an invalid bond in ``info['connectivity']``, or a repeat along
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

    result = atoms.repeat(reps)
    result.info.pop(GraphAttr.SMILES, None)
    if GraphAttr.CONNECTIVITY in atoms.info:
        result.info[GraphAttr.CONNECTIVITY] = _tile_bonds(atoms, reps)
    if original_index is not None:
        tiled = [int(k) for k in original_index] * int(np.prod(reps))
        result.info[NodeAttr.ORIGINAL_INDEX] = tiled
    return result
