import pytest
from rdkit import Chem

import molify


def test_match_substructure():
    atoms = molify.smiles2atoms("CC(=O)O")
    mol = molify.ase2rdkit(atoms, suggestions=["CC(=O)O"])

    # match CH3 fragment (carbon with 3 hydrogens)
    match = molify.match_substructure(mol, "[C;H3]", hydrogens="include")
    assert match == ((0, 4, 5, 6),)
    assert atoms[match[0]].get_chemical_symbols() == ["C", "H", "H", "H"]


def test_match_substructure_box():
    atoms = molify.smiles2conformers("CC(=O)O", 1)
    box = molify.pack([atoms], counts=[3], density=0.5)
    mol = molify.ase2rdkit(box, suggestions=["CC(=O)O"])

    # match CH3 fragment (carbon with 3 hydrogens)
    match = molify.match_substructure(mol, "[C;H3]", hydrogens="include")
    assert match == ((0, 4, 5, 6), (8, 12, 13, 14), (16, 20, 21, 22))
    for m in match:
        assert box[m].get_chemical_symbols() == ["C", "H", "H", "H"]


def test_get_substructure():
    atoms = molify.smiles2atoms("C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]")
    # match NO3 group using smarts
    frames = molify.get_substructures(
        atoms,
        "[N+](=O)[O-]",
        suggestions=["C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]"],
    )
    assert len(frames) == 3
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]


def test_get_substructure_box():
    atoms = molify.smiles2conformers(
        "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", 1
    )
    box = molify.pack([atoms], counts=[3], density=0.5)
    # match NO3 group using smarts
    frames = molify.get_substructures(
        box,
        "[N+](=O)[O-]",
        suggestions=["C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]"],
    )
    assert len(frames) == 9
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]


@pytest.mark.parametrize("remove_connectivity", [True, False])
def test_iter_fragments(remove_connectivity):
    water = molify.smiles2conformers("O", 1)
    box = molify.pack([water], [10], density=500)
    if remove_connectivity:
        del box.info["connectivity"]
    fragments = list(molify.iter_fragments(box))
    assert len(fragments) == 10
    for atoms in fragments:
        assert len(atoms) == 3


def test_iter_fragments_extxyz(extxyz_roundtrip):
    water = molify.smiles2conformers("O", 1)
    ethanol = molify.smiles2conformers("CCO", 1)
    box = extxyz_roundtrip(molify.pack([water, ethanol], [3, 2], density=900))

    fragments = list(molify.iter_fragments(box))

    assert [f.get_chemical_formula() for f in fragments] == ["H2O"] * 3 + ["C2H6O"] * 2


def test_iter_fragments_keeps_unbonded_atoms():
    water = molify.smiles2conformers("O", 1)
    lithium = molify.smiles2conformers("[Li+]", 1)
    box = molify.pack([water, lithium], [1, 1], density=900)

    fragments = list(molify.iter_fragments(box))

    assert [f.get_chemical_formula() for f in fragments] == ["H2O", "Li"]


def test_bmim_bf4_no_info():
    bmim = molify.smiles2conformers("CCCCN1C=C[N+](=C1)C", numConfs=1)
    bf4 = molify.smiles2conformers("[B-](F)(F)(F)F", numConfs=1)

    box = molify.pack(
        [bmim, bf4],
        counts=[10, 10],
        density=500,
        tolerance=3,
    )
    del box.info["connectivity"]

    # Convert to RDKit Mol with suggestions for proper bond detection
    # Without connectivity, we need suggestions to detect aromatic bonds correctly
    mol = molify.ase2rdkit(box, suggestions=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"])

    bf4_matches = molify.match_substructure(mol, "[B-](F)(F)(F)F")
    assert len(bf4_matches) == 10
    for match in bf4_matches:
        assert box[match].get_chemical_symbols() == bf4[0].get_chemical_symbols()

    # Note: RDKit perceives the imidazolium ring as aromatic, so we use
    # lowercase notation
    bmim_matches = molify.match_substructure(
        mol, "CCCCn1cc[n+](C)c1", hydrogens="include"
    )
    assert len(bmim_matches) == 10
    for match in bmim_matches:
        assert len(match) == 25
        assert sorted(box[match].get_chemical_symbols()) == sorted(
            bmim[0].get_chemical_symbols()
        )

    # Also test with explicit hydrogen SMARTS pattern
    # Note: patterns with explicit [H] should use default hydrogens="exclude"
    # since the [H] atoms are part of the pattern itself
    bmim_matches_smarts = molify.match_substructure(
        mol,
        "[H]c1c([H])[n+](C([H])([H])[H])c([H])n1C([H])([H])C([H])([H])C([H])([H])C([H])([H])[H]",
        hydrogens="exclude",
    )
    assert len(bmim_matches_smarts) == 10
    for match in bmim_matches_smarts:
        # When pattern has explicit [H], only heavy atoms are returned by default
        assert len(match) == 10

    bmim_matches_smarts = molify.match_substructure(
        mol,
        "[H]c1c([H])[n+](C([H])([H])[H])c([H])n1C([H])([H])C([H])([H])C([H])([H])C([H])([H])[H]",
        hydrogens="isolated",
    )
    assert len(bmim_matches_smarts) == 10
    for match in bmim_matches_smarts:
        # explicit only hydrogens are returned
        assert len(match) == 15

    bmim_matches_smarts = molify.match_substructure(
        mol,
        "[H]c1c([H])[n+](C([H])([H])[H])c([H])n1C([H])([H])C([H])([H])C([H])([H])C([H])([H])[H]",
        hydrogens="include",
    )
    assert len(bmim_matches_smarts) == 10
    for match in bmim_matches_smarts:
        # explicit only hydrogens are returned
        assert len(match) == 25  # 10 heavy + 15 H


@pytest.fixture
def ethanol_mol():
    """Returns an RDKit molecule for ethanol (CCO) with explicit hydrogens."""
    mol = Chem.MolFromSmiles("CCO")
    return Chem.AddHs(mol)


@pytest.fixture
def toluene_mol():
    """Returns an RDKit molecule for toluene (Cc1ccccc1) with explicit hydrogens."""
    mol = Chem.MolFromSmiles("Cc1ccccc1")
    return Chem.AddHs(mol)


# =============================================================================
# Test Cases for match_substructure with hydrogen handling
# =============================================================================


def test_select_carbons(ethanol_mol):
    """Test selecting all carbon atoms."""
    # Ethanol (CCO with Hs): C(0), C(1), O(2), H(3-8)
    matches = molify.match_substructure(ethanol_mol, "[#6]")
    # Flatten to get all matched indices
    indices = [idx for match in matches for idx in match]
    assert sorted(indices) == [0, 1]


def test_select_oxygen(ethanol_mol):
    """Test selecting the oxygen atom."""
    matches = molify.match_substructure(ethanol_mol, "[#8]")
    indices = [idx for match in matches for idx in match]
    assert sorted(indices) == [2]


def test_no_matches(ethanol_mol):
    """Test a SMARTS pattern that has no matches."""
    matches = molify.match_substructure(ethanol_mol, "[F]")  # Fluorine
    assert matches == ()


# --- Hydrogen Handling Tests ---


def test_hydrogens_excluded_by_default(ethanol_mol):
    """Test that hydrogens are excluded by default."""
    # C-O bond involves atoms 1 and 2. Hydrogens attached are not included.
    matches = molify.match_substructure(ethanol_mol, "CO")
    assert matches == ((1, 2),)


def test_hydrogens_included(ethanol_mol):
    """Test the 'include' option for hydrogens."""
    # C-O bond (atoms 1, 2) plus attached hydrogens (atoms 6, 7, 8)
    # H on O is atom 8. Hs on C(1) are 6, 7.
    matches = molify.match_substructure(ethanol_mol, "CO", hydrogens="include")
    # Expected: C(1), H(6), H(7), O(2), H(8)
    assert matches == ((1, 6, 7, 2, 8),)


def test_hydrogens_isolated(ethanol_mol):
    """Test the 'isolated' option for hydrogens."""
    # Select ONLY the hydrogens from the C-O match
    matches = molify.match_substructure(ethanol_mol, "CO", hydrogens="isolated")
    # Expected: H(6), H(7), H(8)
    assert matches == ((6, 7, 8),)


def test_smarts_with_explicit_hydrogens(ethanol_mol):
    """Test a SMARTS pattern that explicitly includes hydrogens."""
    # Find oxygen attached to hydrogen (pattern includes both O and H)
    # With default hydrogens="exclude", we get only the heavy atom (oxygen)
    matches_exclude = molify.match_substructure(ethanol_mol, "[#8]-[H]")
    assert matches_exclude == ((2,),)

    # With hydrogens="isolated", we get only the hydrogen from the match
    h_matches = molify.match_substructure(ethanol_mol, "[#8]-[H]", hydrogens="isolated")
    assert h_matches == ((8,),)


# --- Mapped SMILES Tests ---


def test_mapped_smiles(ethanol_mol):
    """Test selecting only mapped atoms using a mapped SMILES pattern."""
    # The pattern "[C:1][C:2]O" matches atoms 0, 1, and 2,
    # but only C:1 and C:2 are mapped.
    # With mapped_only=True, should return only the mapped atoms.
    matches = molify.match_substructure(ethanol_mol, "[C:1][C:2]O", mapped_only=True)
    assert matches == ((0, 1),)


def test_mapped_smiles_with_hydrogens(ethanol_mol):
    """Test mapped SMILES with hydrogen filtering."""
    # Pattern "C[O:1]" matches atoms C(1) and O(2), but only O(2) is mapped.
    # With mapped_only=True, core selection will be just atom 2.

    # Include hydrogens attached to the mapped oxygen
    matches_included = molify.match_substructure(
        ethanol_mol, "C[O:1]", hydrogens="include", mapped_only=True
    )
    # Expected: O(2) and its hydrogen H(8)
    assert matches_included == ((2, 8),)

    # Exclude hydrogens (returns just the mapped heavy atom)
    matches_excluded = molify.match_substructure(
        ethanol_mol, "C[O:1]", hydrogens="exclude", mapped_only=True
    )
    assert matches_excluded == ((2,),)

    # Isolate only hydrogens attached to the mapped oxygen
    matches_isolated = molify.match_substructure(
        ethanol_mol, "C[O:1]", hydrogens="isolated", mapped_only=True
    )
    assert matches_isolated == ((8,),)


# --- Error Handling Tests ---


def test_invalid_smarts_raises_error():
    """Test that an invalid SMARTS string raises a ValueError."""
    mol = Chem.MolFromSmiles("C")
    with pytest.raises(ValueError, match="Invalid SMARTS/SMILES"):
        molify.match_substructure(mol, "this is not valid")


# =============================================================================
# Test Cases for group_matches_by_fragment
# =============================================================================


def test_group_matches_by_fragment(ethanol_water):
    """Test grouping matches from disconnected fragments."""
    # ethanol_water is an ase.Atoms object with 2 ethanol and 2 water molecules.
    mol = molify.ase2rdkit(ethanol_water)

    # Test 1: Select all carbon atoms.
    # Should find 4 matches (2 per ethanol)
    matches = molify.match_substructure(mol, "[#6]")
    assert len(matches) == 4

    # Group by fragment - should find 2 groups (the two ethanol molecules)
    grouped = molify.group_matches_by_fragment(mol, matches)
    assert grouped == [[0, 1], [9, 10]]

    # Verify they are indeed carbons
    for group in grouped:
        for idx in group:
            assert mol.GetAtomWithIdx(idx).GetAtomicNum() == 6

    # Test 2: Select all oxygen atoms.
    matches_oxygen = molify.match_substructure(mol, "[#8]")
    grouped_oxygen = molify.group_matches_by_fragment(mol, matches_oxygen)
    assert grouped_oxygen == [[2], [11], [18], [21]]

    # Verify they are indeed oxygens
    for group in grouped_oxygen:
        for idx in group:
            assert mol.GetAtomWithIdx(idx).GetAtomicNum() == 8

    # Test 3: Select a non-existent atom.
    matches_fluorine = molify.match_substructure(mol, "[F]")
    grouped_fluorine = molify.group_matches_by_fragment(mol, matches_fluorine)
    assert grouped_fluorine == []


def test_use_label_multiple_times(alanine_dipeptide):
    """Test selecting atoms using the same label multiple times."""
    mol = molify.ase2rdkit(alanine_dipeptide)
    with pytest.raises(ValueError, match="Atom map label '1' is used multiple times"):
        # we do not allow the same label to be used multiple times for selection,
        # need to be unique
        molify.match_substructure(mol, "CC(=O)N[C:1]([C:1])C(=O)NC", mapped_only=True)


def test_match_substructure_mapped_order(alanine_dipeptide):
    """Test that mapped atoms are returned in map number order."""
    mol = molify.ase2rdkit(alanine_dipeptide)

    # Select single mapped atom
    matches = molify.match_substructure(mol, "CC(=O)N[C:1](C)C(=O)NC", mapped_only=True)
    assert matches == ((4,),)

    matches = molify.match_substructure(mol, "CC(=O)NC([C:1])C(=O)NC", mapped_only=True)
    assert matches == ((5,),)

    matches = molify.match_substructure(mol, "CC(=O)NC(C)[C:1](=O)NC", mapped_only=True)
    assert matches == ((6,),)

    matches = molify.match_substructure(mol, "CC(=O)NC(C)C(=O)[N:1]C", mapped_only=True)
    assert matches == ((8,),)

    # now all of them in order
    matches = molify.match_substructure(
        mol, "CC(=O)N[C:1]([C:2])[C:3](=O)[N:4]C", mapped_only=True
    )
    assert matches == ((4, 5, 6, 8),)

    # now in a different order (should still return in map number order: 1,2,3,4)
    matches = molify.match_substructure(
        mol, "CC(=O)N[C:4]([C:3])[C:2](=O)[N:1]C", mapped_only=True
    )
    # Map numbers are 4,3,2,1 so should return atoms in order: 8,6,5,4
    # Actually, atoms are sorted by map number, so 1,2,3,4 -> 8,6,5,4
    assert matches == ((8, 6, 5, 4),)

    # now with hydrogens which are in order:
    # atom 1, hydrogens of 1, atom 2, hydrogens of 2, ...
    matches = molify.match_substructure(
        mol, "CC(=O)N[C:1](C)C(=O)NC", hydrogens="include", mapped_only=True
    )
    assert matches == ((4, 14),)

    matches = molify.match_substructure(
        mol, "CC(=O)N[C:1]([C:2])C(=O)NC", hydrogens="include", mapped_only=True
    )
    assert matches == ((4, 14, 5, 15, 16, 17),)

    # now inverse order
    matches = molify.match_substructure(
        mol, "CC(=O)N[C:2]([C:1])C(=O)NC", hydrogens="include", mapped_only=True
    )
    assert matches == ((5, 15, 16, 17, 4, 14),)

    # now with hydrogens isolated
    matches = molify.match_substructure(
        mol, "CC(=O)N[C:1]([C:2])C(=O)NC", hydrogens="isolated", mapped_only=True
    )
    assert matches == ((14, 15, 16, 17),)

    matches = molify.match_substructure(
        mol, "CC(=O)N[C:2]([C:1])C(=O)NC", hydrogens="isolated", mapped_only=True
    )
    assert matches == ((15, 16, 17, 14),)


def test_match_substructure_mapped_order_box(alanine_dipeptide_box):
    """Test mapped atom selection with multiple molecules."""
    mol = molify.ase2rdkit(alanine_dipeptide_box)

    matches = molify.match_substructure(mol, "CC(=O)N[C:1](C)C(=O)NC", mapped_only=True)
    # Group by fragment
    grouped = molify.group_matches_by_fragment(mol, matches)
    assert grouped == [[4], [26], [48]]

    matches = molify.match_substructure(
        mol, "CC(=O)N[C:1]([C:2])[C:3](=O)[N:4]C", mapped_only=True
    )
    grouped = molify.group_matches_by_fragment(mol, matches)
    assert grouped == [[4, 5, 6, 8], [26, 27, 28, 30], [48, 49, 50, 52]]

    # Different order
    matches = molify.match_substructure(
        mol, "CC(=O)N[C:4]([C:3])[C:2](=O)[N:1]C", mapped_only=True
    )
    grouped = molify.group_matches_by_fragment(mol, matches)
    assert grouped == [[8, 6, 5, 4], [30, 28, 27, 26], [52, 50, 49, 48]]

    # With hydrogens
    matches = molify.match_substructure(
        mol, "CC(=O)N[C:1](C)C(=O)NC", hydrogens="include", mapped_only=True
    )
    grouped = molify.group_matches_by_fragment(mol, matches)
    assert grouped == [[4, 14], [26, 36], [48, 58]]

    matches = molify.match_substructure(
        mol, "CC(=O)N[C:1]([C:2])C(=O)NC", hydrogens="include", mapped_only=True
    )
    grouped = molify.group_matches_by_fragment(mol, matches)
    assert grouped == [
        [4, 14, 5, 15, 16, 17],
        [26, 36, 27, 37, 38, 39],
        [48, 58, 49, 59, 60, 61],
    ]


def test_match_substructure_ions_with_none_bond_orders():
    """Test that ase2rdkit automatically handles None bond orders.

    This reproduces the exact error scenario from hillclimber where
    connectivity exists but has None bond orders, causing:
    ValueError: Edge (0, 1) is missing 'bond_order' attribute.

    The fix: ase2rdkit now automatically detects and handles this.
    """
    # Create a system with water and ions (reproduces user's scenario)
    water = molify.smiles2conformers("O", numConfs=1)
    na_ion = molify.smiles2conformers("[Na+]", numConfs=1)
    cl_ion = molify.smiles2conformers("[Cl-]", numConfs=1)

    # Pack them together
    box = molify.pack(
        [water, na_ion, cl_ion],
        counts=[16, 1, 1],
        density=1000,
        tolerance=2.0,
    )

    # Simulate corrupted connectivity with None bond orders
    # This causes the exact error the user reported
    box.info["connectivity"] = [(i, j, None) for i, j, _ in box.info["connectivity"]]

    # ase2rdkit should NOW automatically handle this (after our fix)
    mol = molify.ase2rdkit(box)

    # Verify match_substructure works
    na_matches = molify.match_substructure(mol, "[Na+]")
    assert len(na_matches) == 1

    cl_matches = molify.match_substructure(mol, "[Cl-]")
    assert len(cl_matches) == 1

    water_matches = molify.match_substructure(mol, "[OH2]")
    assert len(water_matches) == 16


# =============================================================================
# Test Cases for visualize_selected_molecules
# =============================================================================


def test_visualize_selected_molecules_basic(ethanol_mol):
    """Test basic visualization functionality."""
    # Select some atoms to highlight
    a = [0, 1]  # Carbons
    b = [2]  # Oxygen

    img = molify.visualize_selected_molecules(ethanol_mol, a, b)
    assert img is not None


def test_visualize_selected_molecules_empty_selections(ethanol_mol):
    """Test visualization with empty selections - shows molecule without highlights."""
    img = molify.visualize_selected_molecules(ethanol_mol)
    assert img is not None


def test_visualize_selected_molecules_overlapping_selections(ethanol_mol):
    """Test visualization with overlapping selections (later args take precedence)."""
    a = [0, 1, 2]  # All heavy atoms
    b = [2]  # Oxygen (should get color from b, not a)

    img = molify.visualize_selected_molecules(ethanol_mol, a, b)
    assert img is not None


def test_visualize_selected_molecules_single_selection(ethanol_mol):
    """Test visualization with a single selection."""
    a = [0, 1]  # Carbons

    img = molify.visualize_selected_molecules(ethanol_mol, a)
    assert img is not None


def test_visualize_selected_molecules_multiple_selections(ethanol_mol):
    """Test visualization with multiple selections."""
    a = [0]  # First carbon
    b = [1]  # Second carbon
    c = [2]  # Oxygen

    img = molify.visualize_selected_molecules(ethanol_mol, a, b, c)
    assert img is not None


def test_visualize_selected_molecules_with_alpha(ethanol_mol):
    """Test visualization with custom alpha value."""
    a = [0, 1]  # Carbons
    b = [2]  # Oxygen

    # Test with different alpha values
    img_transparent = molify.visualize_selected_molecules(ethanol_mol, a, b, alpha=0.2)
    assert img_transparent is not None

    img_opaque = molify.visualize_selected_molecules(ethanol_mol, a, b, alpha=1.0)
    assert img_opaque is not None
