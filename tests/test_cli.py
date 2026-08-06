import ase.io
import pytest
from typer.testing import CliRunner

from molify import __version__
from molify.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "smiles2atoms" in result.output


@pytest.mark.parametrize("fmt", ["XYZ", "xyz", "extxyz"])
def test_smiles2atoms_stdout(fmt):
    result = runner.invoke(app, ["smiles2atoms", "CCO", "--format", fmt])
    assert result.exit_code == 0
    lines = result.output.strip().split("\n")
    assert lines[0] == "9"
    assert "smiles=CCO" in lines[1]
    assert "connectivity=" in lines[1]


def test_smiles2atoms_stdout_defaults_to_extxyz():
    result = runner.invoke(app, ["smiles2atoms", "CCO"])
    assert result.exit_code == 0
    assert "smiles=CCO" in result.output


def test_smiles2atoms_output_file(tmp_path):
    path = tmp_path / "etoh.xyz"
    result = runner.invoke(app, ["smiles2atoms", "CCO", "--output", str(path)])
    assert result.exit_code == 0

    atoms = ase.io.read(path)
    assert atoms.get_chemical_formula() == "C2H6O"
    assert atoms.info["smiles"] == "CCO"
    assert len(atoms.info["connectivity"]) == 8


def test_smiles2atoms_format_from_suffix(tmp_path):
    path = tmp_path / "etoh.pdb"
    result = runner.invoke(app, ["smiles2atoms", "CCO", "-o", str(path)])
    assert result.exit_code == 0
    assert path.read_text().startswith("MODEL")
    assert ase.io.read(path).get_chemical_formula() == "C2H6O"


def test_smiles2atoms_format_overrides_suffix(tmp_path):
    path = tmp_path / "etoh.xyz"
    result = runner.invoke(
        app, ["smiles2atoms", "CCO", "-o", str(path), "-f", "proteindatabank"]
    )
    assert result.exit_code == 0
    assert path.read_text().startswith("MODEL")


def test_smiles2atoms_seed_changes_positions(tmp_path):
    positions = []
    for seed in (42, 1234):
        path = tmp_path / f"conf_{seed}.xyz"
        result = runner.invoke(
            app, ["smiles2atoms", "CCO", "-o", str(path), "--seed", str(seed)]
        )
        assert result.exit_code == 0
        positions.append(ase.io.read(path).get_positions())

    assert not (positions[0] == positions[1]).all()


def test_smiles2atoms_invalid_smiles():
    result = runner.invoke(app, ["smiles2atoms", "C(C"])
    assert result.exit_code == 2
    assert "C(C" in result.output


def test_smiles2atoms_unknown_format(tmp_path):
    result = runner.invoke(
        app, ["smiles2atoms", "CCO", "-o", str(tmp_path / "etoh.zzz")]
    )
    assert result.exit_code != 0
