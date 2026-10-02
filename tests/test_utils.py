import networkx as nx
import numpy as np
import pytest
import rdkit.Chem
from ase import Atoms

import molify
from molify.utils import (
    rdkit_determine_bonds,
    suggestions2networkx,
    unwrap_structures,
)


class SMILES:
    PF6: str = "F[P-](F)(F)(F)(F)F"
    Li: str = "[Li+]"
    EC: str = "C1COC(=O)O1"
    EMC: str = "CCOC(=O)OC"


@pytest.fixture
def ec_emc_li_pf6():
    atoms_pf6 = molify.smiles2conformers(SMILES.PF6, numConfs=10)
    atoms_li = molify.smiles2conformers(SMILES.Li, numConfs=10)
    atoms_ec = molify.smiles2conformers(SMILES.EC, numConfs=10)
    atoms_emc = molify.smiles2conformers(SMILES.EMC, numConfs=10)

    return molify.pack(
        data=[atoms_pf6, atoms_li, atoms_ec, atoms_emc],
        counts=[3, 3, 8, 12],
        density=1400,
    )


@pytest.mark.parametrize("scale", [1.0, 0.5, 2.0])
def test_unwrap_structures(scale):
    hexane = molify.smiles2conformers("CCCCCC", numConfs=1)
    box = molify.pack(
        data=[hexane],
        counts=[10],
        density=800,
    )
    shifted_box = box.copy()
    shifted_box.set_positions(
        shifted_box.get_positions() + scale * shifted_box.get_cell().diagonal()
    )
    shifted_box.wrap()  # Wrap back into PBC

    # check that the movement broke some bonds
    off = 0
    graph = molify.ase2networkx(shifted_box)
    assert len(list(nx.connected_components(graph))) == 10
    for i, j in graph.edges():
        off += bool(shifted_box.get_distance(i, j, mic=False) > 1.6)
    assert off > 0

    # check the initial box has no broken bonds
    off = 0
    graph = molify.ase2networkx(box)
    assert len(list(nx.connected_components(graph))) == 10
    for i, j in graph.edges():
        off += bool(box.get_distance(i, j, mic=False) > 1.6)
    assert off == 0

    # Unwrap the structures and assert that bonds are restored
    unwrapped_atoms = unwrap_structures(shifted_box)
    graph = molify.ase2networkx(unwrapped_atoms)
    for i, j in graph.edges():
        assert unwrapped_atoms.get_distance(i, j, mic=False) < 1.6

    # assert edges stay the same / indices remain the same
    graph_unwrapped = molify.ase2networkx(unwrapped_atoms)
    graph_box = molify.ase2networkx(box)
    assert nx.is_isomorphic(graph_unwrapped, graph_box)


@pytest.mark.parametrize("scale", [1.0, 0.5, 2.0])
def test_unwrap_ring_molecules(scale):
    # Create a single benzene conformer
    benzene = molify.smiles2conformers("c1ccccc1", numConfs=1)

    # Pack a box with 10 benzene molecules
    box = molify.pack(
        data=[benzene],
        counts=[10],
        density=800,
    )

    # Copy and shift box to break molecules across PBC
    shifted_box = box.copy()
    shifted_box.set_positions(
        shifted_box.get_positions() + scale * shifted_box.get_cell().diagonal()
    )
    shifted_box.wrap()  # Wraps back into the periodic box
    # NOTE: the wrap seems to break bonds already!
    # and yes, packmol places 2 hydrogens across the PBC boundary!

    # Verify that some bonds are broken (distances > 1.6 Å without MIC)
    graph = molify.ase2networkx(shifted_box)
    assert len(list(nx.connected_components(graph))) == 10  # Sanity check
    broken_bonds = 0
    for i, j in graph.edges():
        d = shifted_box.get_distance(i, j, mic=False)
        if d > 1.6:
            broken_bonds += 1
    assert broken_bonds > 0

    # Verify that unwrapping restores proper bonding geometry
    unwrapped = unwrap_structures(shifted_box)
    graph_unwrapped = molify.ase2networkx(unwrapped)
    assert len(list(nx.connected_components(graph_unwrapped))) == 10

    for i, j in graph_unwrapped.edges():
        d = unwrapped.get_distance(i, j, mic=False)
        assert d < 1.6


def make_molecule(shift=(0, 0, 0), cell=(10, 10, 10), pbc=True):
    """Simple diatomic molecule with optional shift and PBC wrapping."""
    pos = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])  # bond length 1.1
    pos += shift
    atoms = Atoms("H2", positions=pos, cell=cell, pbc=pbc)
    return atoms


def test_no_pbc():
    """Ensure no changes when system has no PBC."""
    atoms = make_molecule()
    atoms.set_pbc(False)

    atoms_unwrapped = unwrap_structures(atoms.copy())
    dist = atoms_unwrapped.get_distance(0, 1)

    assert np.isclose(dist, 1.1, atol=0.05)


def test_idempotent():
    """Calling unwrap on an already unwrapped structure should do nothing."""
    atoms = make_molecule(shift=(0, 0, 0))
    unwrapped = unwrap_structures(atoms.copy())
    unwrapped2 = unwrap_structures(unwrapped.copy())

    assert np.allclose(unwrapped.get_positions(), unwrapped2.get_positions())


@pytest.mark.parametrize(
    "smiles",
    [
        # Simple neutral molecules
        "O",  # Water
        "CC",  # Ethane
        "C1CCCCC1",  # Cyclohexane
        "C1=CC=CC=C1",  # Benzene
        "C1=CC=CC=C1O",  # Phenol
        # Simple anions/cations
        "[Li+]",  # Lithium ion
        "[Na+]",  # Sodium ion
        "[Cl-]",  # Chloride
        "[OH-]",  # Hydroxide
        "[NH4+]",  # Ammonium
        "[CH3-]",  # Methyl anion
        "[C-]#N",  # Cyanide anion"
        # Phosphate and sulfate groups
        "OP(=O)(O)O ",  # H3PO4
        "OP(=O)(O)[O-]",  # H2PO4-
        "[O-]P(=O)(O)[O-]",  # HPO4 2-
        "[O-]P(=O)([O-])[O-]",  # PO4 3-
        "OP(=O)=O",  # HPO3
        "[O-]P(=O)=O",  # PO3 -
        "OS(=O)(=O)O",  # H2SO4
        "OS(=O)(=O)[O-]",  # HSO4-
        "[O-]S(=O)(=O)[O-]",  # SO4 2-
        "[O-]S(=O)(=O)([O-])",  # SO3 2-
        # Multiply charged ions
        # "[Fe+3]",            # Iron(III)
        # "[Fe++]",            # Iron(II) alternative syntax
        # "[O-2]",             # Oxide dianion
        # "[Mg+2]",            # Magnesium ion
        # "[Ca+2]",            # Calcium ion
        # Charged organic fragments
        "C[N+](C)(C)C",  # Tetramethylammonium
        # "[N-]=[N+]=[N-]",       # Azide ion
        "C1=[N+](C=CC=C1)[O-]",  # Nitrobenzene
        # Complex anions
        "F[B-](F)(F)F",  # Tetrafluoroborate
        "F[P-](F)(F)(F)(F)F",  # Hexafluorophosphate
        "[O-]C(=O)C(=O)[O-]",  # Oxalate dianion
        # Zwitterions
        "C(C(=O)[O-])N",  # Glycine
        "C1=CC(=CC=C1)[N+](=O)[O-]",  # Nitrobenzene
        # Aromatic heterocycles
        "c1ccncc1",  # Pyridine
        "c1cccnc1",  # Pyrimidine
        # Polyaromatics
        "c1ccc2ccccc2c1",  # Naphthalene
        "c1ccc2c(c1)ccc3c2cccc3",  # Phenanthrene
    ],
)
def test_rdkit_determine_bonds(smiles: str):
    atoms = molify.smiles2atoms(smiles)
    del atoms.info["connectivity"]
    del atoms.info["smiles"]
    mol = rdkit_determine_bonds(atoms)

    target_mol = rdkit.Chem.MolFromSmiles(smiles)
    target_mol = rdkit.Chem.AddHs(target_mol)
    target_smiles = rdkit.Chem.MolToSmiles(target_mol)
    found_smiles = rdkit.Chem.MolToSmiles(mol)

    assert target_smiles == found_smiles


def test_suggestions2networkx():
    graphs = suggestions2networkx(smiles=["C", "C=O"])
    assert len(graphs) == 2
    assert graphs[0].number_of_nodes() == 5
    assert graphs[0].number_of_edges() == 4

    assert graphs[1].number_of_nodes() == 4  # pbc and cell
    assert graphs[1].number_of_edges() == 3
