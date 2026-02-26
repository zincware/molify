import numpy.testing as npt
import pytest

xyzgraph = pytest.importorskip("xyzgraph")

import molify


def test_ase2networkx_xyzgraph_engine_water():
    """xyzgraph engine produces correct graph for water."""
    atoms = molify.smiles2atoms("O")
    atoms.info.pop("connectivity")

    graph = molify.ase2networkx(atoms, engine="xyzgraph")

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2

    for node_id, data in graph.nodes(data=True):
        assert "atomic_number" in data
        assert "position" in data
        assert "charge" in data
        assert isinstance(data["atomic_number"], int)
        assert isinstance(data["charge"], float)

    for u, v, data in graph.edges(data=True):
        assert "bond_order" in data
        assert data["bond_order"] is not None
        assert data["bond_order"] == 1.0


def test_ase2networkx_xyzgraph_engine_ethanol():
    """xyzgraph engine produces correct graph for ethanol."""
    atoms = molify.smiles2atoms("CCO")
    atoms.info.pop("connectivity")

    graph = molify.ase2networkx(atoms, engine="xyzgraph")

    assert graph.number_of_nodes() == 9
    assert graph.number_of_edges() == 8

    for u, v, data in graph.edges(data=True):
        assert data["bond_order"] is not None


def test_ase2networkx_xyzgraph_engine_formaldehyde():
    """xyzgraph engine detects double bond in formaldehyde."""
    atoms = molify.smiles2atoms("C=O")
    atoms.info.pop("connectivity")

    graph = molify.ase2networkx(atoms, engine="xyzgraph")

    assert graph.number_of_nodes() == 4

    co_bond_order = None
    for u, v, data in graph.edges(data=True):
        nums = {graph.nodes[u]["atomic_number"], graph.nodes[v]["atomic_number"]}
        if nums == {6, 8}:
            co_bond_order = data["bond_order"]
    assert co_bond_order == 2.0


def test_ase2networkx_xyzgraph_engine_preserves_pbc_cell():
    """xyzgraph engine preserves pbc and cell graph attributes."""
    atoms = molify.smiles2atoms("O")
    atoms.info.pop("connectivity")

    graph = molify.ase2networkx(atoms, engine="xyzgraph")

    assert "pbc" in graph.graph
    assert "cell" in graph.graph


def test_ase2networkx_xyzgraph_charge_parameter():
    """Explicit charge parameter is forwarded to xyzgraph."""
    atoms = molify.smiles2atoms("[OH-]")
    atoms.info.pop("connectivity")

    graph = molify.ase2networkx(atoms, engine="xyzgraph", charge=-1)

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1

    total_charge = sum(data["charge"] for _, data in graph.nodes(data=True))
    assert total_charge == pytest.approx(-1.0, abs=0.1)


def test_ase2networkx_xyzgraph_engine_kwargs():
    """engine_kwargs are forwarded to xyzgraph.build_graph."""
    atoms = molify.smiles2atoms("O")
    atoms.info.pop("connectivity")

    graph = molify.ase2networkx(atoms, engine="xyzgraph", quick=True)
    assert graph.number_of_nodes() == 3


def test_ase2networkx_xyzgraph_engine_empty_atoms():
    """xyzgraph engine handles empty atoms gracefully."""
    import ase

    atoms = ase.Atoms()
    graph = molify.ase2networkx(atoms, engine="xyzgraph")
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_ase2networkx_rdkit_engine_unchanged():
    """engine='rdkit' preserves current behavior exactly."""
    atoms = molify.smiles2atoms("O")
    atoms.info.pop("connectivity")

    graph = molify.ase2networkx(atoms, engine="rdkit")

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2

    for u, v, data in graph.edges(data=True):
        assert data["bond_order"] is None


def test_ase2rdkit_xyzgraph_engine_water():
    """ase2rdkit with xyzgraph engine produces correct RDKit molecule for water."""
    from rdkit import Chem

    atoms = molify.smiles2atoms("O")
    atoms.info.pop("connectivity")

    mol = molify.ase2rdkit(atoms, engine="xyzgraph")

    assert mol.GetNumAtoms() == 3
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles("O")), canonical=True
    )


def test_ase2rdkit_xyzgraph_engine_ethanol():
    """ase2rdkit with xyzgraph engine produces correct molecule for ethanol."""
    from rdkit import Chem

    atoms = molify.smiles2atoms("CCO")
    atoms.info.pop("connectivity")

    mol = molify.ase2rdkit(atoms, engine="xyzgraph")

    assert mol.GetNumAtoms() == 9
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles("CCO")), canonical=True
    )


def test_ase2rdkit_xyzgraph_charge_forwarded():
    """ase2rdkit forwards charge parameter to xyzgraph engine."""
    from rdkit import Chem

    atoms = molify.smiles2atoms("[OH-]")
    atoms.info.pop("connectivity")

    mol = molify.ase2rdkit(atoms, engine="xyzgraph", charge=-1)
    assert mol.GetNumAtoms() == 2


def test_ase2rdkit_xyzgraph_engine_formaldehyde():
    """ase2rdkit with xyzgraph engine correctly identifies double bonds."""
    from rdkit import Chem

    atoms = molify.smiles2atoms("C=O")
    atoms.info.pop("connectivity")

    mol = molify.ase2rdkit(atoms, engine="xyzgraph")

    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles("C=O")), canonical=True
    )


def test_ase2networkx_xyzgraph_importerror():
    """engine='xyzgraph' raises ImportError when xyzgraph is not installed."""
    from unittest.mock import patch

    atoms = molify.smiles2atoms("O")
    atoms.info.pop("connectivity")

    with patch("molify.ase2x._xyzgraph", None):
        with pytest.raises(ImportError, match="xyzgraph is required"):
            molify.ase2networkx(atoms, engine="xyzgraph")


def test_ase2networkx_auto_engine_no_xyzgraph():
    """engine='auto' falls back to rdkit behavior when xyzgraph is not installed."""
    from unittest.mock import patch

    atoms = molify.smiles2atoms("O")
    atoms.info.pop("connectivity")

    with patch("molify.ase2x._xyzgraph", None):
        graph = molify.ase2networkx(atoms, engine="auto")

    # Should work fine with rdkit fallback
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    # rdkit path has bond_order=None
    for u, v, data in graph.edges(data=True):
        assert data["bond_order"] is None


def test_ase2networkx_connectivity_takes_precedence_over_engine():
    """When connectivity is present in atoms.info, engine parameter is ignored."""
    atoms = molify.smiles2atoms("O")  # Has connectivity in info

    # Even with engine="xyzgraph", connectivity should be used
    graph = molify.ase2networkx(atoms, engine="xyzgraph")

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    # Bond orders come from connectivity (not None)
    for u, v, data in graph.edges(data=True):
        assert data["bond_order"] is not None
