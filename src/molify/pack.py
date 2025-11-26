import logging
import pathlib
import subprocess
import tempfile
import typing as t

import ase.io
import numpy as np
from ase.io.proteindatabank import write_proteindatabank
from rdkit import Chem

from molify.utils import calculate_box_dimensions

log = logging.getLogger(__name__)

OBJ_OR_STR = t.Union[str, Chem.rdchem.Mol, ase.Atoms]
OBJ_OR_STR_OR_LIST = t.Union[OBJ_OR_STR, t.List[t.Tuple[OBJ_OR_STR, float]]]
FORMAT = t.Literal["pdb", "xyz"]


def _select_conformers(
    data: list[list[ase.Atoms]], counts: list[int], seed: int
) -> list[ase.Atoms]:
    """Randomly selects the required number of conformers for each molecule type."""
    rng = np.random.default_rng(seed)
    selected_images = []
    for images, count in zip(data, counts):
        indices = rng.choice(range(len(images)), count, replace=len(images) < count)
        selected_images.extend([images[idx] for idx in indices])
    return selected_images


def _generate_packmol_input(
    selected_images: list[ase.Atoms],
    cell: list[float],
    tolerance: float,
    seed: int,
    output_format: FORMAT,
    pbc: bool,
) -> str:
    """Generates the input string for the PACKMOL program."""
    packmol_input = f"""
tolerance {tolerance}
filetype {output_format}
output mixture.{output_format}
seed {seed}
"""
    if pbc:
        packmol_input += f"""
pbc 0 0 0 {" ".join([f"{x:.6f}" for x in cell])}
"""
    for i, _ in enumerate(selected_images):
        packmol_input += f"""
structure struct_{i}.{output_format}
    filetype {output_format}
    number 1
    inside box 0. 0. 0. {cell[0]} {cell[1]} {cell[2]}
end structure
"""
    return packmol_input


def _run_packmol(
    packmol_executable: pathlib.Path | str,
    input_file: pathlib.Path,
    tmpdir: pathlib.Path,
    verbose: bool,
) -> None:
    """Executes the PACKMOL program."""
    with open(input_file, "rb") as fin:
        subprocess.run(
            [str(packmol_executable)],
            cwd=tmpdir,
            check=True,
            capture_output=not verbose,
            stdin=fin,
        )


def _write_molecule_files(
    selected_images: list[ase.Atoms], tmpdir: pathlib.Path, _format: FORMAT
) -> None:
    """Writes the individual molecule structures to files in the temporary directory."""
    for i, atoms in enumerate(selected_images):
        filepath = tmpdir / f"struct_{i}.{_format}"
        if _format == "pdb":
            write_proteindatabank(filepath, atoms)
        elif _format == "xyz":
            ase.io.write(filepath, atoms)


def _extract_atom_arrays(
    selected_images: list[ase.Atoms], packed_atoms: ase.Atoms
) -> ase.Atoms:
    """Extracts and adds relevant atom arrays (if present)

    Add bonds from the input structures to the packed structure if available.

    Parameters
    ----------
    selected_images : list[ase.Atoms]
        List of input ASE Atoms objects.
    packed_atoms : ase.Atoms
        The ASE Atoms object representing the packed system.

    Returns
    -------
    ase.Atoms
        The packed ASE Atoms object with the copied arrays and bonds.
    """
    array_keys = [
        "occupancy",
        "bfactor",
        "residuenames",
        "atomtypes",
        "residuenumbers",
    ]
    for key in array_keys:
        if any(key in atoms.arrays for atoms in selected_images):
            if key in packed_atoms.arrays:
                continue
            all_arrays = [atoms.arrays.get(key) for atoms in selected_images]
            concatenated_array = np.concatenate(
                [arr for arr in all_arrays if arr is not None]
            )
            packed_atoms.arrays[key] = concatenated_array

    if all("connectivity" in atoms.info for atoms in selected_images):
        bonds = []
        offset = 0
        for atoms in selected_images:
            for bond in atoms.info["connectivity"]:
                bonds.append((bond[0] + offset, bond[1] + offset, bond[2]))
            offset += len(atoms)
        packed_atoms.info["connectivity"] = bonds
    charges = np.concatenate([atom.get_initial_charges() for atom in selected_images])
    if any(charge != 0 for charge in charges):
        packed_atoms.set_initial_charges(charges)
    return packed_atoms


def pack(
    data: list[list[ase.Atoms]],
    counts: list[int],
    density: float,
    seed: int = 42,
    tolerance: float = 2,
    verbose: bool = False,
    packmol: str = "packmol",
    pbc: bool = True,
    output_format: FORMAT = "pdb",
    ratio: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ase.Atoms:
    """
    Packs the given molecules into a box with the specified density using PACKMOL.

    Parameters
    ----------
    data : list[list[ase.Atoms]]
        A list of lists of ASE Atoms objects representing the molecules to be packed.
    counts : list[int]
        A list of integers representing the number of each type of molecule.
    density : float
        The target density of the packed system in kg/m^3.
    seed : int, optional
        The random seed for reproducibility, by default 42.
    tolerance : float, optional
        The tolerance for the packing algorithm, by default 2.
    verbose : bool, optional
        If True, enables logging of the packing process, by default False.
    packmol : str, optional
        The path to the packmol executable.
    pbc : bool, optional
        Ensure tolerance across periodic boundaries, by default True.
    output_format : str, optional
        The file format used for communication with packmol, by default "pdb".
        WARNING: Do not use "xyz". This might cause issues and
        is only implemented for debugging purposes.
    ratio : tuple[float, float, float], optional
        Box aspect ratio (a:b:c). Must be three positive, finite numbers.
        Defaults to (1.0, 1.0, 1.0) for a cubic box.

    Returns
    -------
    ase.Atoms
        An ASE Atoms object representing the packed system.

    Example
    -------
    >>> from molify import pack, smiles2conformers
    >>> water = smiles2conformers("O", 1)
    >>> ethanol = smiles2conformers("CCO", 1)
    >>> density = 1000  # kg/m^3
    >>> packed_system = pack([water, ethanol], [7, 5], density)
    >>> print(packed_system)
    Atoms(symbols='C10H44O12', pbc=True, cell=[8.4, 8.4, 8.4])
    """
    selected_images = _select_conformers(data, counts, seed)
    cell = calculate_box_dimensions(images=selected_images, density=density)

    # Adjust cell dimensions according to ratio while keeping volume unchanged
    original_volume = np.prod(cell)
    ratio_product = np.prod(ratio)
    scale = original_volume ** (1 / 3) / (ratio_product ** (1 / 3))
    cell = [float(scale * r) for r in ratio]

    packmol_input = _generate_packmol_input(
        selected_images, cell, tolerance, seed, output_format, pbc
    )

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)
        _write_molecule_files(selected_images, tmpdir, output_format)
        (tmpdir / "pack.inp").write_text(packmol_input)

        if verbose:
            print(f"{packmol} < ")
            print(packmol_input)
        _run_packmol(packmol, tmpdir / "pack.inp", tmpdir, verbose)
        try:
            packed_atoms: ase.Atoms = ase.io.read(tmpdir / f"mixture.{output_format}")
            packed_atoms.arrays.pop("atomtypes", None)
            packed_atoms.arrays.pop("bfactor", None)
            packed_atoms.arrays.pop("occupancy", None)
            packed_atoms.arrays.pop("residuenames", None)
            packed_atoms.arrays.pop("residuenumbers", None)
        except FileNotFoundError as e:
            log.error("Packmol Input:\n%s", packmol_input)
            log.error("Using Packmol executable at: %s", packmol)
            log.exception("Packmol did not produce mixture.%s", output_format)
            raise FileNotFoundError(
                f"Packmol did not produce 'mixture.{output_format}'."
                " Please check the input parameters."
            ) from e

    packed_atoms.cell = cell
    packed_atoms.pbc = True
    packed_atoms = _extract_atom_arrays(selected_images, packed_atoms)
    return packed_atoms
