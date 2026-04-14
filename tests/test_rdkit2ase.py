import ase
import numpy as np
import numpy.testing as npt
import pytest
import rdkit
from rdkit import Chem

from molify import ase2rdkit, rdkit2ase, rdkit2networkx, smiles2atoms
from molify.constants import NodeAttr


@pytest.fixture
def methane():
    positions = [
        (0.000, 0.000, 0.000),  # Carbon
        (0.629, 0.629, 0.629),  # Hydrogen
        (-0.629, -0.629, 0.629),  # Hydrogen
        (-0.629, 0.629, -0.629),  # Hydrogen
        (0.629, -0.629, -0.629),  # Hydrogen
    ]
    return ase.Atoms("CH4", positions=positions)


@pytest.fixture
def nacn():
    positions = [
        (0.000, 0.000, 0.000),  # Sodium
        (0.000, 0.000, 1.000),  # Carbon
        (0.000, 0.000, 2.000),  # Nitrogen
    ]
    symbols = ["Na", "C", "N"]
    atoms = ase.Atoms(symbols, positions=positions)
    atoms.set_initial_charges([1, 0, -1])
    return atoms


@pytest.fixture
def methanol():
    return rdkit.Chem.MolFromSmiles("CO")


def test_rdkit2ase(methanol):
    ase_atoms = rdkit2ase(methanol)
    assert isinstance(ase_atoms, ase.Atoms)
    assert ase_atoms.get_chemical_formula() == "CH4O"
    assert ase_atoms.info["smiles"] == "CO"

    assert ase_atoms.info["connectivity"] == [
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (0, 4, 1.0),
        (1, 5, 1.0),
    ]
    assert ase_atoms.get_atomic_numbers().tolist() == [6, 8, 1, 1, 1, 1]


def test_ase2rdkit_methane(methane):
    rdkit_mol = ase2rdkit(methane, suggestions=[])

    # assert the smiles string
    smiles = rdkit.Chem.MolToSmiles(rdkit_mol)
    assert smiles == "[H]C([H])([H])[H]"


def test_ase2rdkit_methane_suggestions(methane):
    rdkit_mol = ase2rdkit(methane, suggestions=["C", "CCC"])

    # assert the smiles string
    smiles = rdkit.Chem.MolToSmiles(rdkit_mol)
    assert smiles == "[H]C([H])([H])[H]"


# TODO: this is currently not working!
# def test_ase2rdkit_nacn(nacn):
#     rdkit_mol = ase2rdkit(nacn)
#     assert isinstance(rdkit_mol, rdkit.Chem.Mol)
#     assert rdkit.Chem.MolToSmiles(rdkit_mol) == "[Na+].[C-]#N"
#     assert rdkit_mol.GetNumAtoms() == 3
#     assert rdkit_mol.GetNumBonds() == 1
#     assert rdkit_mol.GetBondWithIdx(0).GetBondTypeAsDouble() == 3.0


def test_ase2rdkit2ase(methane):
    mol = ase2rdkit(methane, suggestions=[])
    atoms = rdkit2ase(mol)
    assert isinstance(atoms, ase.Atoms)
    assert atoms.get_chemical_formula() == "CH4"


def test_rdkit2ase2rdkit(methanol):
    atoms = rdkit2ase(methanol)
    mol = ase2rdkit(atoms)
    assert isinstance(mol, rdkit.Chem.Mol)
    assert rdkit.Chem.MolToSmiles(rdkit.Chem.RemoveHs(mol)) == "CO"


@pytest.mark.parametrize(
    "smiles, formula",
    [
        ("CCO", "C2H6O"),
        ("CC(=O)O", "C2H4O2"),
        ("C1=CC=CC=C1", "C6H6"),
        ("C1=CC=CC=C1C", "C7H8"),
        ("C1=CC=CC=C1CC", "C8H10"),
        ("C1=CC=CC=C1CCC", "C9H12"),
    ],
)
def test_smiles_to_atoms(smiles, formula):
    atoms = smiles2atoms(smiles)
    assert isinstance(atoms, ase.Atoms)
    assert atoms.get_chemical_formula() == formula
    assert atoms.info["smiles"] == smiles
    assert "initial_charges" not in atoms.arrays


def test_seeded_smiles_to_atoms():
    atoms = smiles2atoms("CCO", seed=42)
    npt.assert_almost_equal(
        atoms.get_positions(),
        [
            [-0.9534551842756259, 0.047804216096192946, 0.04249937663911893],
            [0.48791868768310137, -0.32153931807429764, -0.18904973635963],
            [1.276977526346879, 0.3247836784511733, 0.7376732425087382],
            [-1.3570126123500137, 0.7339751954162012, -0.7332225613847279],
            [-1.593320940326173, -0.8625698312047074, 0.021759475665733226],
            [-1.0710312763466532, 0.5964393608027696, 1.0111627605054987],
            [0.7874060542693432, -0.1406086817099794, -1.235774323901826],
            [0.5900278787458259, -1.426109660842922, -0.04403048576653677],
            [1.8324898662533213, 1.0478250410655707, 0.38898225209363174],
        ],
    )
    # now test with different seeds
    assert not np.allclose(
        atoms.get_positions(), smiles2atoms("CCO", seed=43).get_positions()
    )


def test_rdkit2ase_nacn():
    # create sodium cyanide molecule
    mol = Chem.MolFromSmiles("[C-]#N.[Na+]")
    mol = Chem.AddHs(mol)
    atoms = rdkit2ase(mol)
    assert atoms.get_chemical_formula() == "CNNa"
    assert atoms.info["smiles"] == "[C-]#N.[Na+]"
    assert atoms.info["connectivity"] == [
        (0, 1, 3.0),
    ]
    assert atoms.get_initial_charges().tolist() == [-1, 0, 1]
    assert atoms.get_atomic_numbers().tolist() == [6, 7, 11]


# ============================================================================
# Tests for original_index preservation (GitHub issue #104)
# ============================================================================


class TestRdkit2AseOriginalIndex:
    """Test that rdkit2ase preserves original_index from RDKit atom properties."""

    def test_original_index_carried_from_rdkit_to_ase(self):
        """If RDKit atoms have original_index, store it."""
        mol = Chem.RWMol(Chem.MolFromSmiles("CO"))
        mol = Chem.AddHs(mol)
        # Set original_index on each atom (simulating a prior networkx2rdkit conversion)
        for atom in mol.GetAtoms():
            atom.SetIntProp(NodeAttr.ORIGINAL_INDEX, atom.GetIdx() + 10)

        atoms = rdkit2ase(mol)

        assert NodeAttr.ORIGINAL_INDEX in atoms.info
        expected = [atom.GetIdx() + 10 for atom in mol.GetAtoms()]
        assert atoms.info[NodeAttr.ORIGINAL_INDEX] == expected

    def test_no_original_index_when_rdkit_atoms_lack_property(self):
        """No original_index added when absent from RDKit atoms."""
        mol = Chem.MolFromSmiles("CO")
        atoms = rdkit2ase(mol)

        assert NodeAttr.ORIGINAL_INDEX not in atoms.info


class TestRdkit2NetworkxOriginalIndex:
    """Test that rdkit2networkx preserves original_index from RDKit atom properties."""

    def test_original_index_carried_from_rdkit_to_networkx(self):
        """If RDKit atoms have original_index, use it."""
        mol = Chem.RWMol(Chem.MolFromSmiles("CO"))
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            atom.SetIntProp(NodeAttr.ORIGINAL_INDEX, atom.GetIdx() + 20)

        graph = rdkit2networkx(mol)

        for node_id, attributes in graph.nodes(data=True):
            assert attributes[NodeAttr.ORIGINAL_INDEX] == node_id + 20

    def test_falls_back_to_getidx_without_property(self):
        """Falls back to GetIdx() without original_index."""
        mol = Chem.MolFromSmiles("CO")

        graph = rdkit2networkx(mol)

        for node_id, attributes in graph.nodes(data=True):
            assert attributes[NodeAttr.ORIGINAL_INDEX] == node_id
