import itertools
import re
from functools import partial

import numpy as np
import pytest
from rdkit import Chem

from molify import ase2rdkit, smiles2atoms, smiles2conformers


def test_smiles2conformers():
    smiles = "CCO"
    conformers = smiles2conformers(smiles, numConfs=10)
    assert len(conformers) == 10
    for a, b in itertools.combinations(conformers, 2):
        assert np.sum(np.abs(a.positions - b.positions)) > 0.1


@pytest.mark.parametrize("smiles", ("CCO", "C1=CC=CC=C1", "C1CCCC1"))
def test_smiles_info(smiles):
    """Test that the SMILES string is stored in the info dictionary."""
    conformers = smiles2conformers(smiles, numConfs=10)
    for atoms in conformers:
        assert atoms.info["smiles"] == smiles


@pytest.mark.parametrize("use_numpy", (True, False))
@pytest.mark.parametrize(
    ("smiles", "connectivity"),
    [
        ("O", [(0, 1, 1.0), (0, 2, 1.0)]),
        ("C=O", [(0, 1, 2.0), (0, 2, 1.0), (0, 3, 1.0)]),
        ("C#C", [(0, 1, 3.0), (0, 2, 1.0), (1, 3, 1.0)]),
        (
            "c1ccccc1",  # equal to C1=CC=CC=C1
            [
                (0, 1, 1.5),
                (1, 2, 1.5),
                (2, 3, 1.5),
                (3, 4, 1.5),
                (4, 5, 1.5),
                (5, 0, 1.5),
                (0, 6, 1.0),
                (1, 7, 1.0),
                (2, 8, 1.0),
                (3, 9, 1.0),
                (4, 10, 1.0),
                (5, 11, 1.0),
            ],
        ),
    ],
)
def test_connectivity_info(smiles, connectivity, use_numpy):
    """Test that the connectivity information is stored in the info dictionary."""
    conformers = smiles2conformers(smiles, numConfs=10)

    if use_numpy:
        for atoms in conformers:
            atoms.info["connectivity"] = np.array(atoms.info["connectivity"])

    for atoms in conformers:
        if use_numpy:
            assert np.array_equal(atoms.info["connectivity"], connectivity)
        else:
            assert atoms.info["connectivity"] == connectivity

    # assert smiles backwards
    mol = ase2rdkit(conformers[0])
    # remove hydrogens, compute smiles and compare
    mol = Chem.RemoveHs(mol)
    smiles2 = Chem.MolToSmiles(mol)
    assert smiles2 == smiles


@pytest.mark.parametrize("smiles", ["C1CC", "xx", "C(C)(C)(C)(C)C"])
def test_smiles2conformers_rejects_unparseable_smiles(smiles):
    with pytest.raises(
        ValueError, match=re.escape(f"rdkit cannot parse SMILES {smiles!r}")
    ):
        smiles2conformers(smiles, numConfs=3)


@pytest.mark.parametrize(
    ("convert", "num_confs"),
    [(partial(smiles2conformers, numConfs=3), 3), (smiles2atoms, 1)],
    ids=["smiles2conformers", "smiles2atoms"],
)
def test_rejects_unembeddable_smiles(convert, num_confs):
    smiles = "[Fe](C)(C)(C)(C)(C)(C)(C)(C)"
    message = f"rdkit embedded 0 of {num_confs} conformers of {smiles!r}"
    with pytest.raises(ValueError, match=re.escape(message)):
        convert(smiles)


def test_smiles2conformers_rejects_partial_embedding():
    smiles = "C1CC2CCC1C2"
    assert len(smiles2conformers(smiles, numConfs=10)) == 10
    with pytest.raises(
        ValueError, match=r"^rdkit embedded [1-9] of 10 conformers of 'C1CC2CCC1C2'$"
    ):
        smiles2conformers(smiles, numConfs=10, maxAttempts=1)
