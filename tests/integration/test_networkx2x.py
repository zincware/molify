import numpy.testing as npt
import pytest
from rdkit import Chem

import molify


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


def normalize_connectivity(connectivity):
    """Normalize bond tuples to ensure consistent comparison."""
    return sorted((min(i, j), max(i, j), order) for i, j, order in connectivity)


@pytest.fixture
def graph_smiles_atoms(request):
    smiles = request.param
    atoms = molify.smiles2atoms(smiles)
    return molify.ase2networkx(atoms), smiles, atoms


@pytest.mark.parametrize("graph_smiles_atoms", SMILES_LIST, indirect=True)
def test_networkx2ase(graph_smiles_atoms):
    graph, smiles, atoms = graph_smiles_atoms

    new_atoms = molify.networkx2ase(graph)
    assert new_atoms == atoms

    new_connectivity = normalize_connectivity(new_atoms.info["connectivity"])
    old_connectivity = normalize_connectivity(atoms.info["connectivity"])

    assert new_connectivity == old_connectivity
    npt.assert_array_equal(new_atoms.positions, atoms.positions)
    npt.assert_array_equal(new_atoms.numbers, atoms.numbers)
    npt.assert_array_equal(new_atoms.get_initial_charges(), atoms.get_initial_charges())


def test_networkx2ase_ec_emc_li_pf6(ec_emc_li_pf6):
    atoms = ec_emc_li_pf6
    graph = molify.ase2networkx(atoms)

    new_atoms = molify.networkx2ase(graph)

    new_connectivity = normalize_connectivity(new_atoms.info["connectivity"])
    old_connectivity = normalize_connectivity(atoms.info["connectivity"])

    assert new_atoms == atoms

    assert new_connectivity == old_connectivity
    npt.assert_array_equal(new_atoms.positions, atoms.positions)
    npt.assert_array_equal(new_atoms.numbers, atoms.numbers)
    npt.assert_array_equal(new_atoms.get_initial_charges(), atoms.get_initial_charges())
    npt.assert_array_equal(new_atoms.pbc, atoms.pbc)
    npt.assert_array_equal(new_atoms.cell, atoms.cell)


@pytest.mark.parametrize("graph_smiles_atoms", SMILES_LIST, indirect=True)
def test_networkx2rdkit(graph_smiles_atoms):
    graph, smiles, atoms = graph_smiles_atoms

    mol = molify.networkx2rdkit(graph)
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
    )


def test_networkx2rdkit_ec_emc_li_pf6(ec_emc_li_pf6):
    atoms = ec_emc_li_pf6
    graph = molify.ase2networkx(atoms)
    mol = molify.networkx2rdkit(graph)

    assert len(mol.GetAtoms()) == len(atoms)
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    assert sorted(atom_types) == sorted(atoms.get_atomic_numbers().tolist())

    connectivity = atoms.info["connectivity"]
    assert len(connectivity) == 266
    for i, j, bond_order in connectivity:
        bond = mol.GetBondBetweenAtoms(i, j)
        assert bond is not None
        assert bond.GetBondTypeAsDouble() == bond_order


def test_networkx2ase_subgraph_index_mapping():
    """Test that networkx2ase correctly maps node indices when working with subgraphs.

    This test ensures that when a subgraph is created with non-sequential node indices,
    the resulting ASE atoms object has correctly mapped connectivity information.
    """
    # Create a simple molecule and convert to graph
    atoms = molify.smiles2atoms("CCO")  # Ethanol
    graph = molify.ase2networkx(atoms)

    # Create a subgraph with non-sequential node indices
    # This simulates the scenario that caused the original bug
    subgraph_nodes = [1, 2, 4]  # Non-sequential indices
    subgraph = graph.subgraph(subgraph_nodes).copy()

    # Convert back to ASE atoms
    subgraph_atoms = molify.networkx2ase(subgraph)

    # Check that the atoms object has the correct number of atoms
    assert len(subgraph_atoms) == len(subgraph_nodes)

    # Check connectivity information
    connectivity = subgraph_atoms.info["connectivity"]

    # All indices in connectivity should be valid for the atoms object
    for i, j, bond_order in connectivity:
        assert 0 <= i < len(subgraph_atoms)
        assert 0 <= j < len(subgraph_atoms)
        assert isinstance(bond_order, float)

    # Test that unwrap_structures works correctly with this subgraph
    # (This was the original failing case)
    unwrapped = molify.unwrap_structures(subgraph_atoms)
    assert len(unwrapped) == len(subgraph_atoms)


# ============================================================================
# Tests for networkx2rdkit with None bond orders (new architecture)
# ============================================================================


@pytest.mark.parametrize("graph_smiles_atoms", SMILES_LIST, indirect=True)
def test_networkx2rdkit_with_none_bond_orders(graph_smiles_atoms):
    """Test networkx2rdkit determines bond orders when graph has bond_order=None."""
    graph, smiles, atoms = graph_smiles_atoms

    # Create a graph with bond_order=None by removing bond orders
    graph_no_orders = graph.copy()
    for u, v in graph_no_orders.edges():
        graph_no_orders.edges[u, v]["bond_order"] = None

    # This should determine bond orders automatically
    mol = molify.networkx2rdkit(graph_no_orders, suggestions=[])

    # Verify the molecule is correct
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
    )


@pytest.mark.parametrize("graph_smiles_atoms", SMILES_LIST, indirect=True)
def test_networkx2rdkit_with_smiles_suggestions(graph_smiles_atoms):
    """Test networkx2rdkit uses SMILES suggestions for bond order determination."""
    graph, smiles, atoms = graph_smiles_atoms

    # Create a graph with bond_order=None
    graph_no_orders = graph.copy()
    for u, v in graph_no_orders.edges():
        graph_no_orders.edges[u, v]["bond_order"] = None

    # Use SMILES suggestion for bond order determination
    mol = molify.networkx2rdkit(graph_no_orders, suggestions=[smiles])

    # Verify the molecule matches expected structure
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
    )


def test_networkx2rdkit_preserves_input_graph():
    """Test that networkx2rdkit doesn't modify the input graph."""
    atoms = molify.smiles2atoms("CC")
    graph = molify.ase2networkx(atoms)

    # Remove bond orders
    for u, v in graph.edges():
        graph.edges[u, v]["bond_order"] = None

    # Store original state
    original_edges = {(u, v): data.copy() for u, v, data in graph.edges(data=True)}

    # Call networkx2rdkit
    _ = molify.networkx2rdkit(graph, suggestions=[])

    # Verify original graph is unchanged
    for u, v, data in graph.edges(data=True):
        assert data["bond_order"] is None
        assert original_edges[(u, v)] == data


def test_networkx2rdkit_mixed_bond_orders():
    """Test networkx2rdkit with mixed None and specified bond orders."""
    # Create a simple molecule
    atoms = molify.smiles2atoms("CCO")
    graph = molify.ase2networkx(atoms)
    _ = atoms.info["connectivity"].copy()

    # Set some bond orders to None
    edges = list(graph.edges())
    if len(edges) >= 2:
        graph.edges[edges[0][0], edges[0][1]]["bond_order"] = None

    # This should determine only the None bond orders
    mol = molify.networkx2rdkit(graph, suggestions=[])

    # Verify molecule is correct
    expected_smiles = Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles("CCO")), canonical=True
    )
    assert Chem.MolToSmiles(mol, canonical=True) == expected_smiles


def test_networkx2rdkit_complex_molecule_with_none_bond_orders(ec_emc_li_pf6):
    """Test networkx2rdkit with complex molecule missing bond orders."""
    atoms = ec_emc_li_pf6
    connectivity = atoms.info["connectivity"].copy()
    graph = molify.ase2networkx(atoms)

    # Remove all bond orders
    for u, v in graph.edges():
        graph.edges[u, v]["bond_order"] = None

    # Determine bond orders
    mol = molify.networkx2rdkit(graph, suggestions=[])

    # Verify molecule has correct structure
    assert len(mol.GetAtoms()) == len(atoms)
    assert len(mol.GetBonds()) == len(connectivity)

    # Verify all bonds have determined orders
    for bond in mol.GetBonds():
        assert bond.GetBondTypeAsDouble() > 0


def test_networkx2rdkit_error_on_failed_determination():
    """Test networkx2rdkit raises error when bond determination fails."""
    import networkx as nx

    # Create a pathological graph with impossible geometry
    graph = nx.Graph()
    graph.add_node(0, atomic_number=6, charge=0, position=[0, 0, 0])
    graph.add_node(
        1, atomic_number=6, charge=0, position=[100, 100, 100]
    )  # Too far apart
    graph.add_edge(0, 1, bond_order=None)
    graph.graph["pbc"] = False
    graph.graph["cell"] = None

    # This should raise an error (from underlying bond determination)
    with pytest.raises(ValueError, match="Failed to determine bonds"):
        molify.networkx2rdkit(graph, suggestions=[])


# ============================================================================
# Tests for graph modification workflows (remove/add nodes)
# These tests verify the workflow: ASE -> networkx -> modify -> ASE/rdkit
# ============================================================================


def test_modify_graph_remove_nodes_to_ase():
    """Test ASE -> networkx -> remove nodes -> ASE workflow.

    Verifies that after removing nodes from a networkx graph,
    the resulting ASE atoms object has correct structure.
    """
    # Create ethanol molecule (C-C-O with hydrogens)
    atoms = molify.smiles2atoms("CCO")  # Ethanol
    graph = molify.ase2networkx(atoms)

    original_num_nodes = graph.number_of_nodes()

    # Identify hydrogen nodes to remove (keep only heavy atoms)
    heavy_atom_nodes = [n for n, d in graph.nodes(data=True) if d["atomic_number"] != 1]

    # Create modified graph by removing hydrogen nodes
    modified_graph = graph.subgraph(heavy_atom_nodes).copy()

    # Verify the modified graph has fewer nodes
    assert modified_graph.number_of_nodes() < original_num_nodes
    assert modified_graph.number_of_nodes() == len(heavy_atom_nodes)

    # Convert back to ASE
    new_atoms = molify.networkx2ase(modified_graph)

    # Verify the new atoms object
    assert len(new_atoms) == len(heavy_atom_nodes)
    assert all(num != 1 for num in new_atoms.get_atomic_numbers())

    # Verify connectivity is valid
    connectivity = new_atoms.info["connectivity"]
    for i, j, bond_order in connectivity:
        assert 0 <= i < len(new_atoms)
        assert 0 <= j < len(new_atoms)


def test_modify_graph_remove_nodes_to_rdkit():
    """Test ASE -> networkx -> remove nodes -> rdkit workflow.

    Verifies that after removing nodes from a networkx graph,
    the graph can be converted to an RDKit molecule.

    This test keeps bond orders intact when creating subgraph,
    which is the typical use case when you know the molecular structure.
    """
    # Create ethanol molecule
    atoms = molify.smiles2atoms("CCO")  # Ethanol
    graph = molify.ase2networkx(atoms)

    original_num_nodes = graph.number_of_nodes()

    # Get the oxygen atom and its connected atoms (OH group with one H)
    oxygen_nodes = [n for n, d in graph.nodes(data=True) if d["atomic_number"] == 8]
    assert len(oxygen_nodes) == 1
    oxygen_node = oxygen_nodes[0]

    # Find the H attached to oxygen
    oh_hydrogen = None
    for neighbor in graph.neighbors(oxygen_node):
        if graph.nodes[neighbor]["atomic_number"] == 1:
            oh_hydrogen = neighbor
            break

    # Remove just the OH hydrogen - keeping bond orders from original connectivity
    nodes_to_keep = [n for n in graph.nodes() if n != oh_hydrogen]
    modified_graph = graph.subgraph(nodes_to_keep).copy()

    # Verify the modified graph has one fewer node
    assert modified_graph.number_of_nodes() == original_num_nodes - 1

    # Convert to RDKit - bond orders are preserved from original connectivity
    mol = molify.networkx2rdkit(modified_graph, suggestions=[])

    # Verify the RDKit molecule
    assert mol.GetNumAtoms() == len(nodes_to_keep)

    # Verify the molecule can be sanitized (valid chemistry)
    # Note: removing an H may result in a radical which is still valid
    assert mol.GetNumBonds() >= 1


def test_modify_graph_add_nodes_to_ase():
    """Test ASE -> networkx -> add nodes -> ASE workflow.

    Verifies that after adding nodes to a networkx graph,
    the resulting ASE atoms object has correct structure.
    """
    import numpy as np

    # Create water molecule (O with explicit hydrogens added by smiles2atoms)
    atoms = molify.smiles2atoms("O")  # H2O with 3 atoms
    graph = molify.ase2networkx(atoms)

    original_num_nodes = graph.number_of_nodes()

    # Get the position of the oxygen atom
    oxygen_node = [n for n, d in graph.nodes(data=True) if d["atomic_number"] == 8][0]
    oxygen_pos = graph.nodes[oxygen_node]["position"]

    # Add a new carbon atom bonded to the oxygen (simulating extension)
    new_node_id = max(graph.nodes()) + 1
    new_position = oxygen_pos + np.array([1.4, 0.0, 0.0])  # ~C-O bond distance

    graph.add_node(
        new_node_id,
        position=new_position,
        atomic_number=6,  # Carbon
        original_index=new_node_id,
        charge=0,
    )
    graph.add_edge(oxygen_node, new_node_id, bond_order=1.0)

    # Verify the modified graph has more nodes
    assert graph.number_of_nodes() == original_num_nodes + 1

    # Convert back to ASE
    new_atoms = molify.networkx2ase(graph)

    # Verify the new atoms object
    assert len(new_atoms) == original_num_nodes + 1
    assert 6 in new_atoms.get_atomic_numbers()  # Carbon should be present

    # Verify connectivity includes the new bond
    connectivity = new_atoms.info["connectivity"]
    assert len(connectivity) > 0
    for i, j, bond_order in connectivity:
        assert 0 <= i < len(new_atoms)
        assert 0 <= j < len(new_atoms)


def test_modify_graph_add_nodes_to_rdkit():
    """Test ASE -> networkx -> add nodes -> rdkit workflow.

    Verifies that after adding nodes to a networkx graph,
    the graph can be converted to an RDKit molecule.
    """
    import numpy as np

    # Create methane molecule (C with explicit hydrogens added by smiles2atoms)
    atoms = molify.smiles2atoms("C")  # CH4 with 5 atoms (1 C + 4 H)
    graph = molify.ase2networkx(atoms)
    original_num_atoms = graph.number_of_nodes()

    # Find a carbon atom
    carbon_nodes = [n for n, d in graph.nodes(data=True) if d["atomic_number"] == 6]
    assert len(carbon_nodes) == 1
    carbon_node = carbon_nodes[0]

    # Find a hydrogen bonded to the carbon
    hydrogen_neighbors = [
        n for n in graph.neighbors(carbon_node) if graph.nodes[n]["atomic_number"] == 1
    ]
    assert len(hydrogen_neighbors) > 0
    h_to_replace = hydrogen_neighbors[0]
    h_pos = graph.nodes[h_to_replace]["position"]

    # Remove the hydrogen and add a carbon (making ethane-like structure)
    graph.remove_node(h_to_replace)

    new_node_id = max(graph.nodes()) + 1
    new_carbon_pos = h_pos + np.array([0.5, 0.0, 0.0])

    graph.add_node(
        new_node_id,
        position=new_carbon_pos,
        atomic_number=6,  # Carbon
        original_index=new_node_id,
        charge=0,
    )
    graph.add_edge(carbon_node, new_node_id, bond_order=1.0)

    # Convert to RDKit
    mol = molify.networkx2rdkit(graph, suggestions=[])

    # Verify the RDKit molecule has the expected structure
    # We removed 1 H and added 1 C, so atom count stays the same
    assert mol.GetNumAtoms() == original_num_atoms

    # Should have 2 carbons now (originally 1)
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    assert carbon_count == 2


def test_modify_graph_combined_add_remove_to_ase():
    """Test combined add and remove operations on graph before converting to ASE.

    This tests the full workflow:
    ASE -> networkx -> remove nodes -> add nodes -> ASE
    """
    import numpy as np

    # Create ethanol molecule
    atoms = molify.smiles2atoms("CCO")  # Ethanol
    graph = molify.ase2networkx(atoms)

    original_num_nodes = graph.number_of_nodes()

    # Remove some hydrogen atoms
    hydrogen_nodes = [n for n, d in graph.nodes(data=True) if d["atomic_number"] == 1]
    nodes_to_remove = hydrogen_nodes[:2]  # Remove first 2 hydrogens

    for node in nodes_to_remove:
        graph.remove_node(node)

    # Add a new atom
    remaining_nodes = list(graph.nodes())
    if remaining_nodes:
        ref_node = remaining_nodes[0]
        ref_pos = graph.nodes[ref_node]["position"]
        new_node_id = max(remaining_nodes) + 1

        graph.add_node(
            new_node_id,
            position=ref_pos + np.array([1.5, 0.0, 0.0]),
            atomic_number=7,  # Nitrogen
            original_index=new_node_id,
            charge=0,
        )
        graph.add_edge(ref_node, new_node_id, bond_order=1.0)

    # Convert to ASE
    new_atoms = molify.networkx2ase(graph)

    # Verify structure
    expected_num_atoms = original_num_nodes - 2 + 1  # Removed 2, added 1
    assert len(new_atoms) == expected_num_atoms

    # Verify nitrogen is present
    assert 7 in new_atoms.get_atomic_numbers()

    # Verify connectivity is valid
    connectivity = new_atoms.info["connectivity"]
    for i, j, bond_order in connectivity:
        assert 0 <= i < len(new_atoms)
        assert 0 <= j < len(new_atoms)


def test_modify_graph_combined_add_remove_to_rdkit():
    """Test combined add and remove operations on graph before converting to rdkit.

    This tests the full workflow:
    ASE -> networkx -> remove nodes -> add nodes -> rdkit
    """
    import numpy as np

    # Create methanol molecule
    atoms = molify.smiles2atoms("CO")  # Methanol
    graph = molify.ase2networkx(atoms)

    # Remove some hydrogen atoms
    hydrogen_nodes = [n for n, d in graph.nodes(data=True) if d["atomic_number"] == 1]
    if len(hydrogen_nodes) >= 2:
        nodes_to_remove = hydrogen_nodes[:2]
        for node in nodes_to_remove:
            graph.remove_node(node)

    # Add a fluorine atom to the carbon
    carbon_nodes = [n for n, d in graph.nodes(data=True) if d["atomic_number"] == 6]
    if carbon_nodes:
        carbon_node = carbon_nodes[0]
        carbon_pos = graph.nodes[carbon_node]["position"]
        new_node_id = max(graph.nodes()) + 1

        graph.add_node(
            new_node_id,
            position=carbon_pos + np.array([1.4, 0.0, 0.0]),
            atomic_number=9,  # Fluorine
            original_index=new_node_id,
            charge=0,
        )
        graph.add_edge(carbon_node, new_node_id, bond_order=1.0)

    # Convert to RDKit - use suggestions to help with bond order determination
    mol = molify.networkx2rdkit(graph, suggestions=[])

    # Verify the molecule has the expected atoms
    atom_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    assert 9 in atom_nums  # Fluorine should be present
    assert 6 in atom_nums  # Carbon should be present
    assert 8 in atom_nums  # Oxygen should be present
