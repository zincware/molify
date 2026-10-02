import copy
import io
import re

import ase
import ase.io
import numpy as np
import pytest
from ase.build import bulk
from rdkit import Chem

import molify


def edges(graph) -> set[frozenset[int]]:
    return {frozenset(edge) for edge in graph.edges}


def distance_edges(atoms: ase.Atoms) -> set[frozenset[int]]:
    stripped = atoms.copy()
    stripped.info.pop("connectivity", None)
    stripped.info.pop("original_index", None)
    return edges(molify.ase2networkx(stripped))


def wrapped(atoms: ase.Atoms) -> ase.Atoms:
    shifted = atoms.copy()
    shifted.positions += shifted.cell.sum(axis=0) * 0.5
    shifted.wrap()
    return shifted


def assert_valid_fixture(atoms: ase.Atoms) -> None:
    crossing = [
        (i, j)
        for i, j, _ in atoms.info["connectivity"]
        if atoms.get_distance(int(i), int(j), mic=False) > 2.0
    ]
    assert crossing, "fixture needs bonds across the periodic boundary"
    assert edges(molify.ase2networkx(atoms)) == distance_edges(atoms)


@pytest.fixture(scope="module")
def ethanol_water_box() -> ase.Atoms:
    ethanol = molify.smiles2conformers("CCO", numConfs=10)
    water = molify.smiles2conformers("O", numConfs=10)
    box = molify.pack([ethanol, water], counts=[3, 3], density=700, seed=42)
    return wrapped(box)


@pytest.mark.parametrize("rep", [1, 2, (2, 1, 1), (1, 2, 3)])
def test_repeat_connectivity_matches_distance_bonds(ethanol_water_box, rep):
    box = ethanol_water_box
    assert_valid_fixture(box)

    result = molify.repeat(box, rep)

    n_images = int(np.prod(np.broadcast_to(rep, 3)))
    expected = distance_edges(box.repeat(rep))
    assert edges(molify.ase2networkx(result)) == expected
    assert len(expected) == n_images * len(box.info["connectivity"])


def fragment_smiles(atoms: ase.Atoms) -> list[str]:
    mol = molify.ase2rdkit(atoms)
    return sorted(
        Chem.MolToSmiles(Chem.RemoveHs(frag))
        for frag in Chem.GetMolFrags(mol, asMols=True)
    )


def test_repeat_preserves_bond_orders():
    benzene = molify.smiles2conformers("c1ccccc1", numConfs=3)
    acrylic_acid = molify.smiles2conformers("C=CC(=O)O", numConfs=3)
    box = wrapped(
        molify.pack([benzene, acrylic_acid], counts=[2, 2], density=600, seed=42)
    )
    assert_valid_fixture(box)
    source = fragment_smiles(box)
    assert sorted(set(source)) == ["C=CC(=O)O", "c1ccccc1"]
    n = len(box)
    source_orders = {
        frozenset((i, j)): order for i, j, order in box.info["connectivity"]
    }
    assert set(source_orders.values()) == {1.0, 1.5, 2.0}

    result = molify.repeat(box, (2, 2, 1))

    assert all(
        order == source_orders[frozenset((i % n, j % n))]
        for i, j, order in result.info["connectivity"]
    )
    assert fragment_smiles(result) == sorted(source * 4)


def molecule_at_origin(smiles: str, cell, pbc) -> ase.Atoms:
    atoms = molify.smiles2atoms(smiles)
    atoms.cell = cell
    atoms.pbc = pbc
    atoms.positions -= atoms.get_center_of_mass()
    atoms.wrap()
    return atoms


@pytest.mark.parametrize("rep", [1, 2, (2, 1, 1), (1, 2, 3)])
def test_repeat_triclinic_cell(rep):
    atoms = molecule_at_origin(
        "C=CC(=O)O", cell=[[8.0, 0.0, 0.0], [4.0, 7.5, 0.0], [2.0, 2.0, 7.0]], pbc=True
    )
    assert_valid_fixture(atoms)

    result = molify.repeat(atoms, rep)

    assert edges(molify.ase2networkx(result)) == distance_edges(atoms.repeat(rep))


@pytest.mark.parametrize("rep", [2, (2, 1, 1), (1, 2, 1)])
def test_repeat_sheared_cell(ethanol_water_box, rep):
    box = ethanol_water_box.copy()
    length = box.cell.lengths()[0]
    box.set_cell([[length, 0.0, 0.0], [3 * length, length, 0.0], [0.0, 0.0, length]])
    box = wrapped(box)
    assert_valid_fixture(box)

    result = molify.repeat(box, rep)

    assert edges(molify.ase2networkx(result)) == distance_edges(box.repeat(rep))


def test_repeat_partial_pbc():
    slab = molecule_at_origin("CCO", cell=[8.0, 8.0, 0.0], pbc=[True, True, False])
    assert_valid_fixture(slab)

    result = molify.repeat(slab, (2, 3, 1))

    assert edges(molify.ase2networkx(result)) == distance_edges(slab.repeat((2, 3, 1)))
    with pytest.raises(ValueError, match="undefined lattice vector"):
        molify.repeat(slab, (1, 1, 2))


def test_repeat_pbc_along_zero_cell_vector():
    slab = molecule_at_origin("CCO", cell=[8.0, 8.0, 0.0], pbc=[True, True, False])
    expected = molify.repeat(slab, (2, 3, 1))
    slab.pbc = True

    result = molify.repeat(slab, (2, 3, 1))

    assert result.info["connectivity"] == expected.info["connectivity"]


def test_repeat_tiles_original_index():
    atoms = molecule_at_origin("C=O", cell=[6.0, 6.0, 6.0], pbc=True)
    atoms.info["original_index"] = [5, 6, 7, 8]

    result = molify.repeat(atoms, 2)

    assert result.info["original_index"] == [5, 6, 7, 8] * 8
    graph = molify.ase2networkx(result)
    assert [graph.nodes[k]["original_index"] for k in range(32)] == [5, 6, 7, 8] * 8


def test_repeat_without_connectivity(ethanol_water_box):
    atoms = ethanol_water_box.copy()
    del atoms.info["connectivity"]

    result = molify.repeat(atoms, (2, 1, 1))

    expected = atoms.repeat((2, 1, 1))
    assert "connectivity" not in result.info
    np.testing.assert_array_equal(result.positions, expected.positions)
    np.testing.assert_array_equal(result.numbers, expected.numbers)
    np.testing.assert_array_equal(result.cell, expected.cell)
    np.testing.assert_array_equal(result.pbc, expected.pbc)


def test_repeat_distance_based_connectivity(ethanol_water_box):
    stripped = ethanol_water_box.copy()
    del stripped.info["connectivity"]
    box = molify.networkx2ase(molify.ase2networkx(stripped))
    assert {order for *_, order in box.info["connectivity"]} == {None}

    result = molify.repeat(box, (2, 1, 1))

    assert edges(molify.ase2networkx(result)) == distance_edges(box.repeat((2, 1, 1)))
    assert all(order is None for *_, order in result.info["connectivity"])


def test_repeat_empty_connectivity():
    atoms = ase.Atoms(
        "Ar2", positions=[[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]], cell=[6.0] * 3, pbc=True
    )
    atoms.info["connectivity"] = []

    result = molify.repeat(atoms, (2, 1, 3))

    assert result.info["connectivity"] == []


def test_repeat_leaves_input_unchanged(ethanol_water_box):
    box = ethanol_water_box
    connectivity = copy.deepcopy(box.info["connectivity"])
    positions = box.positions.copy()

    result = molify.repeat(box, 2)

    assert box.info["connectivity"] == connectivity
    np.testing.assert_array_equal(box.positions, positions)
    assert all(
        type(i) is int and type(j) is int for i, j, _ in result.info["connectivity"]
    )


def test_repeat_drops_smiles():
    atoms = molecule_at_origin("CCO", cell=[8.0, 8.0, 8.0], pbc=True)

    result = molify.repeat(atoms, 2)

    assert "smiles" not in result.info
    assert atoms.info["smiles"] == "CCO"


def test_repeat_extxyz_roundtrip():
    atoms = molecule_at_origin(
        "C=CC(=O)O", cell=[[8.0, 0.0, 0.0], [4.0, 7.5, 0.0], [2.0, 2.0, 7.0]], pbc=True
    )
    atoms.info["original_index"] = list(range(10, 10 + len(atoms)))
    buffer = io.StringIO()
    ase.io.write(buffer, atoms, format="extxyz")
    buffer.seek(0)
    loaded = ase.io.read(buffer, format="extxyz")
    assert isinstance(loaded.info["connectivity"], np.ndarray)
    assert isinstance(loaded.info["original_index"], np.ndarray)
    assert_valid_fixture(loaded)

    result = molify.repeat(loaded, 2)

    assert edges(molify.ase2networkx(result)) == distance_edges(loaded.repeat(2))
    assert result.info["original_index"] == list(range(10, 10 + len(atoms))) * 8
    assert all(type(k) is int for k in result.info["original_index"])
    assert all(
        type(i) is int and type(j) is int and type(order) is float
        for i, j, order in result.info["connectivity"]
    )


@pytest.mark.parametrize(
    ("rep", "equivalent"),
    [(np.int64(2), 2), (np.array([2, 1, 1]), (2, 1, 1))],
)
def test_repeat_accepts_numpy_rep(ethanol_water_box, rep, equivalent):
    result = molify.repeat(ethanol_water_box, rep)

    expected = molify.repeat(ethanol_water_box, equivalent)
    assert result.info["connectivity"] == expected.info["connectivity"]


@pytest.mark.parametrize("rep", [0, -1, (0, 1, 1), (2,), (2, 1), (2.0, 1, 1), "2", 2.0])
def test_repeat_invalid_rep(rep):
    atoms = molecule_at_origin("O", cell=[6.0, 6.0, 6.0], pbc=True)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "rep must be a positive integer or a sequence of three positive "
            f"integers, got {rep!r}"
        ),
    ):
        molify.repeat(atoms, rep)


def test_repeat_original_index_length_mismatch():
    atoms = molecule_at_origin("C=O", cell=[6.0, 6.0, 6.0], pbc=True)
    atoms.info["original_index"] = [5, 6, 7]

    with pytest.raises(
        ValueError,
        match=re.escape(
            "atoms.info['original_index'] holds 3 entries for 4 atoms; "
            "it needs one entry per atom"
        ),
    ):
        molify.repeat(atoms, 2)


@pytest.mark.parametrize("bad_bond", [(0, 4, 1.0), (0, -1, 1.0)])
def test_repeat_connectivity_index_out_of_range(bad_bond):
    atoms = molecule_at_origin("C=O", cell=[6.0, 6.0, 6.0], pbc=True)
    atoms.info["connectivity"] = [*atoms.info["connectivity"], bad_bond]

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"bond ({bad_bond[0]}, {bad_bond[1]}) in atoms.info['connectivity'] "
            "needs atom indices in the range 0..3"
        ),
    ):
        molify.repeat(atoms, 2)


def test_repeat_self_bond():
    atoms = molecule_at_origin("C=O", cell=[6.0, 6.0, 6.0], pbc=True)
    atoms.info["connectivity"] = [*atoms.info["connectivity"], (2, 2, 1.0)]

    with pytest.raises(
        ValueError,
        match=re.escape(
            "bond (2, 2) in atoms.info['connectivity'] links atom 2 to itself"
        ),
    ):
        molify.repeat(atoms, 2)


def test_repeat_cell_too_small_for_unique_bond_images():
    silicon = bulk("Si", "diamond", a=5.43)
    silicon.info["connectivity"] = [(0, 1, 1.0)]

    with pytest.raises(
        ValueError,
        match=re.escape(
            "bond (0, 1) in atoms.info['connectivity'] is 2.351 Å long; repeat needs "
            "every bond shorter than 1.920 Å, half the shortest periodic lattice vector"
        ),
    ):
        molify.repeat(silicon, 3)


def test_repeat_is_public():
    assert "repeat" in molify.__all__
