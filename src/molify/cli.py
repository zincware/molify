"""Command line interface for molify."""

import io
import sys
from typing import Annotated

import ase.io
import typer
from ase.io.formats import extension2format

from molify import __version__
from molify import smiles2atoms as _smiles2atoms

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

    Each command mirrors the molify function of the same name and writes the
    structure to standard output.
    """


@app.command()
def smiles2atoms(
    smiles: Annotated[
        str, typer.Argument(help="SMILES string of the molecule, such as 'CCO'.")
    ],
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help=(
                "Output format, given as an ASE format name or file extension."
                " 'XYZ' selects extxyz, which keeps the SMILES string and the"
                " connectivity."
            ),
        ),
    ] = "extxyz",
    seed: Annotated[
        int, typer.Option(help="Random seed for conformer generation.")
    ] = 42,
) -> None:
    """Convert a SMILES string into a single 3D structure on standard output.

    Example: molify smiles2atoms CCO --format XYZ > etoh.xyz
    """
    atoms = _smiles2atoms(smiles, seed=seed)
    with io.StringIO() as handle:
        ase.io.write(handle, atoms, format=_resolve_format(fmt))
        sys.stdout.write(handle.getvalue())
