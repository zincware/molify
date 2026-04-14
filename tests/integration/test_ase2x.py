from unittest.mock import patch

import ase
import pytest
from rdkit import Chem

import molify
from molify.constants import GraphAttr, NodeAttr


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


# Shared test cases
SMILES_LIST = [
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
    # "[O-]P(=O)(O)[O-]",  # HPO4 2-
    # "[O-]P(=O)([O-])[O-]",  # PO4 3-
    "OP(=O)=O",  # HPO3
    # "[O-]P(=O)=O",  # PO3 -
    "OS(=O)(=O)O",  # H2SO4
    "OS(=O)(=O)[O-]",  # HSO4-
    # "[O-]S(=O)(=O)[O-]",  # SO4 2-
    # "[O-]S(=O)(=O)([O-])",  # SO3 2-
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
    # "[O-]C(=O)C(=O)[O-]",  # Oxalate dianion
    # Zwitterions
    "C(C(=O)[O-])N",  # Glycine
    "C1=CC(=CC=C1)[N+](=O)[O-]",  # Nitrobenzene
    # Aromatic heterocycles
    "C1CCNCC1",
    # Polyaromatics
    "C1CCC2CCCCC2C1",  # Naphthalene
    "C1CCC2C(C1)CCC1CCCCC12",  # Phenanthrene
]


@pytest.fixture
def atoms_and_connectivity(request):
    smiles = request.param
    atoms = molify.smiles2atoms(smiles)
    connectivity = atoms.info["connectivity"]
    return smiles, atoms, connectivity


@pytest.mark.parametrize("atoms_and_connectivity", SMILES_LIST, indirect=True)
def test_ase2networkx(atoms_and_connectivity):
    _, atoms, connectivity = atoms_and_connectivity
    graph = molify.ase2networkx(atoms)
    for i, j, bond_order in connectivity:
        assert graph.has_edge(i, j)
        assert graph.edges[i, j]["bond_order"] == bond_order


@pytest.mark.parametrize("atoms_and_connectivity", SMILES_LIST, indirect=True)
def test_ase2networkx_no_connectivity(atoms_and_connectivity):
    _, atoms, connectivity = atoms_and_connectivity
    atoms.info.pop("connectivity")
    graph = molify.ase2networkx(atoms)
    for i, j, bond_order in connectivity:
        assert graph.has_edge(i, j)
        assert graph.edges[i, j]["bond_order"] is None


@pytest.mark.parametrize("atoms_and_connectivity", SMILES_LIST, indirect=True)
def test_ase2rdkit(atoms_and_connectivity):
    smiles, atoms, connectivity = atoms_and_connectivity
    mol = molify.ase2rdkit(atoms)
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
    )

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        if (a1, a2, order) in connectivity:
            pass
        elif (a2, a1, order) in connectivity:
            pass
        else:
            raise AssertionError(
                f"Bond ({a1}, {a2}, {order}) not found in connectivity: {connectivity}"
            )


@pytest.mark.parametrize("atoms_and_connectivity", SMILES_LIST, indirect=True)
def test_ase2rdkit_no_connectivity(atoms_and_connectivity):
    """Test ase2rdkit determines bond orders when connectivity is not provided."""
    smiles, atoms, connectivity = atoms_and_connectivity
    atoms.info.pop("connectivity")
    mol = molify.ase2rdkit(atoms, suggestions=[])
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
    )


@pytest.mark.parametrize("atoms_and_connectivity", SMILES_LIST, indirect=True)
def test_ase2rdkit_smiles_connectivity(atoms_and_connectivity):
    smiles, atoms, connectivity = atoms_and_connectivity
    atoms.info.pop("connectivity")
    mol = molify.ase2rdkit(atoms, suggestions=[smiles])
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
    )


def test_ase2networkx_ec_emc_li_pf6(ec_emc_li_pf6):
    graph = molify.ase2networkx(ec_emc_li_pf6)
    assert len(graph.nodes) == 3 * 7 + 3 * 1 + 15 * 12 + 10 * 8
    connectivity = ec_emc_li_pf6.info["connectivity"]
    for i, j, bond_order in connectivity:
        assert graph.has_edge(i, j)
        assert graph.edges[i, j]["bond_order"] == bond_order

    li_nodes = [d for _, d in graph.nodes(data=True) if d["atomic_number"] == 3]
    assert len(li_nodes) == 3
    for d in li_nodes:
        assert d["charge"] == 1


def test_ase2networkx_ec_emc_li_pf6_no_connectivity(ec_emc_li_pf6):
    connectivity = ec_emc_li_pf6.info.pop("connectivity")
    ec_emc_li_pf6.set_initial_charges()
    graph = molify.ase2networkx(ec_emc_li_pf6)
    for i, j, bond_order in connectivity:
        assert graph.has_edge(i, j)
        assert graph.edges[i, j]["bond_order"] is None

    # There is no charge information if not using suggestions
    # li_nodes = [d for _, d in graph.nodes(data=True) if d["atomic_number"] == 3]
    # assert len(li_nodes) == 3
    # for d in li_nodes:
    #     assert d["charge"] == 1


@pytest.mark.parametrize("smiles", ["[CH3]", "[O]", "C[O]"])
def test_radicals(smiles):
    atoms = molify.smiles2atoms(smiles)
    graph = molify.ase2networkx(atoms)

    mol = Chem.MolFromSmiles(smiles)
    for i, atom in enumerate(mol.GetAtoms()):
        assert graph.nodes[i]["charge"] == atom.GetFormalCharge()


@pytest.mark.parametrize(
    "smiles,expected_charge",
    [("[NH4+]", 1), ("[O-2]", -2), ("[Fe+3]", 3), ("C[N+](C)(C)C", 1)],
)
def test_formal_charges(smiles, expected_charge):
    atoms = molify.smiles2atoms(smiles)
    total_charge = sum(atoms.get_initial_charges())
    assert total_charge == pytest.approx(expected_charge)


def test_ase2networkx_pbc_flag():
    """Test that pbc flag controls periodic boundary conditions in bond detection."""
    # Create a simple periodic system with atoms that would be bonded across PBC
    atoms = ase.Atoms(
        symbols=["H", "H"],
        positions=[[0.0, 0.0, 0.0], [2.9, 0.0, 0.0]],  # Close across PBC
        cell=[3.0, 3.0, 3.0],
        pbc=True,
    )

    # Remove any existing connectivity
    if "connectivity" in atoms.info:
        atoms.info.pop("connectivity")

    # Test with pbc=True (should find bond across periodic boundary)
    graph_pbc_true = molify.ase2networkx(atoms, pbc=True)

    # Test with pbc=False (should not find bond across periodic boundary)
    graph_pbc_false = molify.ase2networkx(atoms, pbc=False)

    # With pbc=True, atoms should be bonded (distance is 0.1 across PBC)
    # With pbc=False, atoms should not be bonded (distance is 2.9 > typical H-H cutoff)
    assert graph_pbc_true.has_edge(0, 1), (
        "With pbc=True, should find bond across periodic boundary"
    )
    assert not graph_pbc_false.has_edge(0, 1), (
        "With pbc=False, should not find bond across periodic boundary"
    )


@patch("molify.ase2x.vesin", None)
def test_ase2networkx_vesin_none():
    """Test that ase2networkx works correctly when vesin is not available."""
    atoms = molify.smiles2atoms("CC")  # Simple ethane molecule
    atoms.info.pop(
        "connectivity"
    )  # Remove connectivity to force neighbor list calculation

    # Should work without vesin and use ASE neighbor_list
    graph = molify.ase2networkx(atoms)

    # Should have the expected number of nodes and at least one edge (C-C bond)
    assert len(graph.nodes) == len(atoms)
    assert len(graph.edges) >= 1


def test_ase2networkx_vesin_available():
    """Test that vesin is available and can be used."""
    try:
        import vesin  # noqa: F401

        vesin_available = True
    except ImportError:
        vesin_available = False

    atoms = molify.smiles2atoms("CC")  # Simple ethane molecule
    atoms.info.pop(
        "connectivity"
    )  # Remove connectivity to force neighbor list calculation

    if vesin_available:
        # Should work with vesin installed
        graph = molify.ase2networkx(atoms, pbc=True)
        assert len(graph.nodes) == len(atoms)
        assert len(graph.edges) >= 1
    else:
        pytest.skip("vesin not available in this environment")


def test_ase2networkx_mixed_pbc_fallback(ethanol_water):
    """Test ase2networkx with mixed PBC - should fall back to ASE when vesin fails."""
    try:
        import vesin  # noqa: F401
    except ImportError:
        raise pytest.skip("vesin not available in this environment")

    # Use ethanol_water fixture but modify PBC to trigger vesin failure
    atoms = ethanol_water.copy()
    # Set PBC to False for one dimension to trigger the issue
    atoms.pbc = [True, False, True]  # Mixed PBC that can cause vesin to fail

    # Remove connectivity to force neighbor list calculation
    atoms.info.pop("connectivity", None)

    # This should work even if vesin fails, thanks to the fallback
    graph = molify.ase2networkx(atoms, pbc=True)
    assert len(graph.nodes) == len(atoms)
    # Should have some edges from the molecules
    assert len(graph.edges) > 0

    # Verify the graph properties
    assert graph.graph["pbc"] is not None
    assert graph.graph["cell"] is not None


def test_ase2rdkit_with_none_bond_orders_in_connectivity():
    """Test the full pipeline handles connectivity with None bond orders.

    This is the specific issue addressed by the refactoring:
    - User has atoms with connectivity information
    - But bond orders are None in that connectivity
    - The pipeline should determine bond orders automatically
    """
    # Create atoms with proper connectivity
    atoms = molify.smiles2atoms("CCO")  # Ethanol
    original_connectivity = atoms.info["connectivity"].copy()

    # Simulate user-provided connectivity with None bond orders
    # This mimics what happens when connectivity is known but bond orders aren't
    connectivity_without_orders = [(i, j, None) for i, j, _ in original_connectivity]
    atoms.info["connectivity"] = connectivity_without_orders

    # This should work seamlessly - ase2networkx preserves connectivity,
    # then networkx2rdkit determines bond orders
    mol = molify.ase2rdkit(atoms, suggestions=[])

    # Verify the molecule is correct
    expected_smiles = Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles("CCO")), canonical=True
    )
    assert Chem.MolToSmiles(mol, canonical=True) == expected_smiles

    # Verify all bonds have determined orders
    for bond in mol.GetBonds():
        assert bond.GetBondTypeAsDouble() > 0


def test_ase2networkx_preserves_none_bond_orders_in_connectivity():
    """Test ase2networkx preserves connectivity even with None bond orders."""
    # Create atoms with connectivity that has None bond orders
    atoms = molify.smiles2atoms("CC")
    original_connectivity = atoms.info["connectivity"].copy()

    # Set bond orders to None
    connectivity_without_orders = [(i, j, None) for i, j, _ in original_connectivity]
    atoms.info["connectivity"] = connectivity_without_orders

    # ase2networkx should preserve this connectivity (with None bond orders)
    graph = molify.ase2networkx(atoms)

    # Verify connectivity is preserved
    for i, j, bond_order in connectivity_without_orders:
        assert graph.has_edge(i, j)
        assert graph.edges[i, j]["bond_order"] is None


class TestAse2NetworkxOriginalIndex:
    """Test that ase2networkx reads back original_index from atoms.info."""

    def test_reads_stored_original_index_with_connectivity(self):
        """ase2networkx should use atoms.info['original_index'] when present.

        This tests the _create_graph_from_connectivity path.
        """
        atoms = molify.smiles2atoms("CCO")
        # Manually set non-trivial original_index so the test isn't trivially true
        atoms.info[NodeAttr.ORIGINAL_INDEX] = list(range(10, 10 + len(atoms)))

        graph = molify.ase2networkx(atoms)

        expected = list(range(10, 10 + len(atoms)))
        actual = [graph.nodes[n][NodeAttr.ORIGINAL_INDEX] for n in graph.nodes]
        assert actual == expected

    def test_reads_stored_original_index_without_connectivity(self):
        """ase2networkx should use atoms.info['original_index'] when present.

        This tests the _add_node_properties path (no connectivity in atoms.info).
        """
        atoms = molify.smiles2atoms("CCO")
        # Store non-trivial original_index, then remove connectivity
        atoms.info[NodeAttr.ORIGINAL_INDEX] = list(range(10, 10 + len(atoms)))
        del atoms.info[GraphAttr.CONNECTIVITY]

        graph = molify.ase2networkx(atoms)

        expected = list(range(10, 10 + len(atoms)))
        actual = [graph.nodes[n][NodeAttr.ORIGINAL_INDEX] for n in graph.nodes]
        assert actual == expected

    def test_defaults_to_atom_index_without_stored_indices(self):
        """Without atoms.info['original_index'], use sequential atom.index."""
        atoms = molify.smiles2atoms("CCO")
        assert NodeAttr.ORIGINAL_INDEX not in atoms.info

        graph = molify.ase2networkx(atoms)

        for n in graph.nodes:
            assert graph.nodes[n][NodeAttr.ORIGINAL_INDEX] == n
