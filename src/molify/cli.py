"""Command line interface for molify."""

import pathlib
from typing import Annotated, Optional

import ase
import ase.io
import typer
from ase.io.formats import extension2format
from rdkit import Chem

from molify import __version__
from molify import smiles2atoms as _smiles2atoms

STDOUT_FORMAT = "extxyz"

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


def _resolve_format(value: str) -> str:
    """Map a format name or file extension onto an ASE format name.

    Extensions resolve first, so ``XYZ`` selects ASE's ``extxyz`` writer, which
    keeps the cell and the ``atoms.info`` payload such as the SMILES string and
    the connectivity.

    Parameters
    ----------
    value : str
        Format name or file extension, in any capitalisation.

    Returns
    -------
    str
        The format name passed on to :func:`ase.io.write`.
    """
    key = value.strip().lower()
    fmt = extension2format.get(key)
    return fmt.name if fmt is not None else key


def _write(
    images: ase.Atoms | list[ase.Atoms],
    fmt: Optional[str],
    output: Optional[pathlib.Path],
) -> None:
    """Write structures to a file, or to standard output.

    Parameters
    ----------
    images : ase.Atoms or list of ase.Atoms
        The structures to write.
    fmt : str or None
        An ASE format name. ``None`` takes the format from the ``output``
        suffix, and ``extxyz`` on standard output.
    output : pathlib.Path or None
        Target file. ``None`` writes to standard output.
    """
    if output is None:
        ase.io.write("-", images, format=fmt or STDOUT_FORMAT)
    elif fmt is None:
        ase.io.write(output, images)
    else:
        ase.io.write(output, images, format=fmt)


def _validate_smiles(value: str) -> str:
    """Confirm that RDKit parses the given SMILES string.

    Parameters
    ----------
    value : str
        The SMILES string to validate.

    Returns
    -------
    str
        The validated SMILES string.

    Raises
    ------
    typer.BadParameter
        If RDKit rejects the SMILES string.
    """
    if Chem.MolFromSmiles(value) is None:
        raise typer.BadParameter(f"RDKit reads valid SMILES, and rejected {value!r}")
    return value


def _version(value: bool) -> None:
    """Print the molify version and exit."""
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version,
            is_eager=True,
            help="Show the molify version and exit.",
        ),
    ] = False,
) -> None:
    """Build molecular structures from the command line.

    Each command mirrors the molify function of the same name.
    """


@app.command()
def smiles2atoms(
    smiles: Annotated[
        str,
        typer.Argument(
            callback=_validate_smiles,
            help="SMILES string of the molecule, such as 'CCO'.",
        ),
    ],
    fmt: Annotated[
        Optional[str],
        typer.Option(
            "--format",
            "-f",
            help=(
                "Output format, given as an ASE format name or file extension."
                " 'XYZ' selects extxyz, which keeps the SMILES string and the"
                " connectivity. Taken from the --output suffix when omitted,"
                " and 'extxyz' on standard output."
            ),
        ),
    ] = None,
    output: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--output",
            "-o",
            help="Write to this file. Standard output when omitted.",
        ),
    ] = None,
    seed: Annotated[
        int, typer.Option(help="Random seed for conformer generation.")
    ] = 42,
) -> None:
    """Convert a SMILES string into a single 3D structure.

    Example: molify smiles2atoms CCO --format XYZ > etoh.xyz
    """
    atoms = _smiles2atoms(smiles, seed=seed)
    _write(atoms, _resolve_format(fmt) if fmt else None, output)
