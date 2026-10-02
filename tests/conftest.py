import io

import ase
import ase.io
import pytest

import molify


@pytest.fixture(scope="session")
def ethanol_water():
    ethanol = molify.smiles2conformers("CCO", numConfs=100)
    water = molify.smiles2conformers("O", numConfs=100)
    box = molify.pack([ethanol, water], counts=[2, 2], density=700)
    return box.copy()


@pytest.fixture(scope="session")
def alanine_dipeptide() -> ase.Atoms:
    """Alanine dipeptide conformer using molify.smiles2conformers."""
    return molify.smiles2atoms("CC(=O)NC(C)C(=O)NC")


@pytest.fixture(scope="session")
def alanine_dipeptide_box(alanine_dipeptide) -> ase.Atoms:
    """Box of alanine dipeptide molecules using molify.pack"""
    box = molify.pack([[alanine_dipeptide]], counts=[3], density=500)
    return box.copy()


@pytest.fixture
def extxyz_roundtrip():
    def roundtrip(atoms: ase.Atoms) -> ase.Atoms:
        buffer = io.StringIO()
        ase.io.write(buffer, atoms, format="extxyz")
        buffer.seek(0)
        return ase.io.read(buffer, format="extxyz")

    return roundtrip
