import ase
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from molify.constants import GraphAttr


def get_pf6() -> ase.Atoms:
    directions = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    )
    p_position = np.zeros(3, dtype=float)
    f_positions = directions * 1.57

    positions = [p_position] + list(f_positions)
    symbols = ["P"] + ["F"] * 6

    atoms = ase.Atoms(symbols=symbols, positions=positions)

    atoms.info[GraphAttr.CONNECTIVITY] = [(0, i, 1.0) for i in range(1, 7)]
    atoms.info[GraphAttr.SMILES] = "F[P-](F)(F)(F)(F)F"
    atoms.set_initial_charges([-1] + [0] * 6)

    return atoms


def smiles2atoms(smiles: str, seed: int = 42) -> ase.Atoms:
    """Convert a SMILES string to an ASE Atoms object.

    Parameters
    ----------
    smiles : str
        The SMILES string to convert.
    seed : int, optional
        Random seed for conformer generation (default is 42).

    Returns
    -------
    ase.Atoms
        The generated Atoms object (first conformer).

    Notes
    -----
    This is a convenience wrapper around smiles2conformers that returns
    just the first conformer.

    Examples
    --------
    >>> from molify import smiles2atoms
    >>> import ase
    >>> atoms = smiles2atoms("CCO")
    >>> isinstance(atoms, ase.Atoms)
    True
    """
    return smiles2conformers(smiles, 1, seed)[0]


def smiles2conformers(
    smiles: str,
    numConfs: int,  # noqa N803
    randomSeed: int = 42,  # noqa N803
    maxAttempts: int = 1000,  # noqa N803
) -> list[ase.Atoms]:
    """Generate multiple molecular conformers from a SMILES string.

    Parameters
    ----------
    smiles : str
        The SMILES string to convert.
    numConfs : int
        Number of conformers to generate.
    randomSeed : int, optional
        Random seed for conformer generation (default is 42).
    maxAttempts : int, optional
        Maximum number of embedding attempts (default is 1000).

    Returns
    -------
    list[ase.Atoms]
        List of generated conformers as ASE Atoms objects.

    Notes
    -----
    Special handling is included for PF6- (hexafluorophosphate) which
    is treated as a special case.

    Each Atoms object in the returned list includes:
    - Atomic positions
    - Atomic numbers
    - SMILES string in the info dictionary
    - Connectivity information in the info dictionary
    - Formal charges when present

    Examples
    --------
    >>> from molify import smiles2conformers
    >>> import ase
    >>> frames = smiles2conformers("CCO", numConfs=3)
    >>> len(frames)
    3
    >>> all(isinstance(atoms, ase.Atoms) for atoms in frames)
    True
    """
    mol = Chem.MolFromSmiles(smiles)
    if Chem.MolToSmiles(mol, canonical=True) == "F[P-](F)(F)(F)(F)F":
        return [get_pf6()] * numConfs

    mol = Chem.AddHs(mol)
    rdDistGeom.EmbedMultipleConfs(
        mol,
        numConfs=numConfs,
        randomSeed=randomSeed,
        maxAttempts=maxAttempts,
    )

    images: list[ase.Atoms] = []
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    if not any(charge != 0 for charge in charges):
        charges = None

    # collect the bond information
    bonds = []
    for bond in mol.GetBonds():
        bonds.append(
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondTypeAsDouble(),
            )
        )

    for conf in mol.GetConformers():
        atoms = ase.Atoms(
            positions=conf.GetPositions(),
            numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
        )
        atoms.info[GraphAttr.SMILES] = smiles
        atoms.info[GraphAttr.CONNECTIVITY] = bonds
        if charges is not None:
            atoms.set_initial_charges(charges)
        images.append(atoms)

    return images
