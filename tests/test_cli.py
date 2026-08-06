import io

import ase.io
import pytest
from typer.testing import CliRunner

from molify import __version__
from molify.cli import app

runner = CliRunner()


def read_output(result) -> ase.Atoms:
    """Read the structure the CLI wrote to standard output."""
    return ase.io.read(io.StringIO(result.output), format="extxyz")


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "smiles2atoms" in result.output


@pytest.mark.parametrize("fmt", ["XYZ", "xyz", "extxyz"])
def test_smiles2atoms_format_aliases(fmt):
    result = runner.invoke(app, ["smiles2atoms", "CCO", "--format", fmt])
    assert result.exit_code == 0

    lines = result.output.strip().split("\n")
    assert lines[0] == "9"
    assert "smiles=CCO" in lines[1]
    assert "connectivity=" in lines[1]


def test_smiles2atoms_defaults_to_extxyz():
    result = runner.invoke(app, ["smiles2atoms", "CCO"])
    assert result.exit_code == 0

    atoms = read_output(result)
    assert atoms.get_chemical_formula() == "C2H6O"
    assert atoms.info["smiles"] == "CCO"
    assert len(atoms.info["connectivity"]) == 8


def test_smiles2atoms_pdb():
    result = runner.invoke(app, ["smiles2atoms", "CCO", "-f", "pdb"])
    assert result.exit_code == 0
    assert result.output.startswith("MODEL")
    assert ase.io.read(io.StringIO(result.output), format="proteindatabank")


def test_smiles2atoms_seed_changes_positions():
    positions = [
        read_output(
            runner.invoke(app, ["smiles2atoms", "CCO", "--seed", str(seed)])
        ).get_positions()
        for seed in (42, 1234)
    ]
    assert not (positions[0] == positions[1]).all()


def test_smiles2atoms_invalid_smiles():
    result = runner.invoke(app, ["smiles2atoms", "C(C"])
    assert result.exit_code == 2
    assert "C(C" in result.output


@pytest.mark.parametrize("fmt", ["zzz", "cif"])
def test_smiles2atoms_ase_rejects_format(fmt):
    result = runner.invoke(app, ["smiles2atoms", "CCO", "-f", fmt])
    assert result.exit_code != 0
