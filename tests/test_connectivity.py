import re

import numpy as np
import pytest
from rdkit.Chem import AddHs, MolFromSmiles, MolToSmiles

import molify

# from molify.connectivity import reconstruct_bonds_from_template


@pytest.mark.parametrize("remove_connectivity", [True, False])
def test_ase2networkx(remove_connectivity):
    atoms = molify.smiles2atoms("CO")
    if remove_connectivity:
        atoms.info.pop("connectivity", None)
    graph = molify.ase2networkx(atoms)
    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 5

    assert graph.nodes[0]["atomic_number"] == 6
    assert graph.nodes[0]["position"].tolist() == pytest.approx(
        [-0.37, 0.0, 0.0], abs=1e-2
    )
    assert graph.nodes[0]["original_index"] == 0
    assert graph.nodes[0]["charge"] == 0

    assert graph.edges[(0, 1)]["bond_order"] in [None, 1]


def test_rdkit2networkx():
    etoh = MolFromSmiles("CCO")
    etoh = AddHs(etoh)
    graph = molify.rdkit2networkx(etoh)
    assert graph.number_of_nodes() == 9
    assert graph.number_of_edges() == 8

    assert graph.nodes[0]["atomic_number"] == 6
    assert graph.nodes[0]["original_index"] == 0
    assert graph.nodes[0]["charge"] == 0


def test_networkx2rdkit():
    atoms = molify.smiles2atoms("CO")
    graph = molify.ase2networkx(atoms)
    mol = molify.networkx2rdkit(graph)
    assert mol.GetNumAtoms() == 6
    assert mol.GetNumBonds() == 5

    # SMILES representation (including hydrogens)
    assert MolToSmiles(mol) == "[H]OC([H])([H])[H]"


def test_networkx2ase():
    atoms = molify.smiles2atoms("CO")
    graph = molify.ase2networkx(atoms)
    new_atoms = molify.networkx2ase(graph)
    assert new_atoms.get_chemical_symbols() == atoms.get_chemical_symbols()
    assert new_atoms.get_positions().tolist() == atoms.get_positions().tolist()
    assert new_atoms.info["connectivity"] == atoms.info["connectivity"]
    assert (
        new_atoms.get_initial_charges().tolist() == atoms.get_initial_charges().tolist()
    )


def test_networkx2ase_numpy_types():
    atoms = molify.smiles2atoms("CO")
    atoms.info["connectivity"] = np.array(atoms.info["connectivity"], dtype=float)
    # this did raise an error before the fix
    molify.unwrap_structures(atoms)


CONNECTIVITY_READERS = {
    "ase2networkx": molify.ase2networkx,
    "repeat": lambda atoms: molify.repeat(atoms, 2),
    "pack": lambda atoms: molify.pack([[atoms]], [1], density=1000),
    "iter_fragments": lambda atoms: list(molify.iter_fragments(atoms)),
    "compress": lambda atoms: molify.compress(atoms, 1000, freeze_molecules=True),
}


@pytest.mark.parametrize(
    "reader", CONNECTIVITY_READERS.values(), ids=CONNECTIVITY_READERS
)
@pytest.mark.parametrize(
    ("bad_bond", "message"),
    [
        pytest.param(
            (0, 4, 1.0),
            "bond (0, 4) in atoms.info['connectivity'] needs atom indices in the "
            "range 0..3",
            id="out-of-range",
        ),
        pytest.param(
            (0, -1, 1.0),
            "bond (0, -1) in atoms.info['connectivity'] needs atom indices in the "
            "range 0..3",
            id="negative",
        ),
        pytest.param(
            (2, 2, 1.0),
            "bond (2, 2) in atoms.info['connectivity'] links atom 2 to itself",
            id="self-bond",
        ),
        pytest.param(
            (0, 1.9, 1.0),
            "bond (0, 1.9) in atoms.info['connectivity'] needs integer atom indices",
            id="non-integer",
        ),
        pytest.param(
            (0, 1),
            "bond (0, 1) in atoms.info['connectivity'] needs three entries "
            "(i, j, order)",
            id="two-entries",
        ),
    ],
)
def test_invalid_connectivity(reader, bad_bond, message):
    atoms = molify.smiles2atoms("C=O")
    atoms.cell = [6.0, 6.0, 6.0]
    atoms.pbc = True
    atoms.info["connectivity"] = [*atoms.info["connectivity"], bad_bond]

    with pytest.raises(ValueError, match=re.escape(message)):
        reader(atoms)
