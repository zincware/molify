import typing as tp

import ase
import matplotlib.pyplot as plt
from ase.build import separate
from rdkit import Chem
from rdkit.Chem import Draw

from molify.ase2x import ase2rdkit
from molify.constants import GraphAttr
from molify.utils import find_connected_components


def match_substructure(  # noqa: C901
    mol: Chem.Mol,
    smarts_or_smiles: str,
    hydrogens: tp.Literal["include", "exclude", "isolated"] = "exclude",
    mapped_only: bool = False,
) -> tuple[tuple[int, ...], ...]:
    """Find all matches of a substructure pattern in an RDKit molecule.

    This function performs substructure matching with advanced features including
    hydrogen handling and atom mapping support. All matches are returned, including
    matches across multiple disconnected fragments.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule to search. Use ``molify.ase2rdkit(atoms, suggestions=[...])``
        to convert from ASE Atoms with explicit control over bond detection.
    smarts_or_smiles : str
        SMARTS pattern or mapped SMILES pattern (e.g., "[C:1][C:2]O").
        If atom maps are present and ``mapped_only=True``, only mapped atoms
        are returned in map number order.
    hydrogens : {'exclude', 'include', 'isolated'}, default='exclude'
        Controls hydrogen atom inclusion in matches:

        - 'exclude': Return only heavy atoms (default)
        - 'include': Return heavy atoms followed by their bonded hydrogens
        - 'isolated': Return only hydrogens bonded to matched heavy atoms
    mapped_only : bool, default=False
        If True and pattern contains atom maps (e.g., [C:1]), return only the
        mapped atoms ordered by map number. If False, return all matched atoms.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Tuple of matches, where each match is a tuple of atom indices.
        For patterns with multiple matches, each match is returned as a separate tuple.

    Raises
    ------
    ValueError
        If the SMARTS/SMILES pattern is invalid or if atom map labels are used
        multiple times within the same pattern.

    Examples
    --------
    >>> from molify import smiles2atoms, ase2rdkit, match_substructure
    >>>
    >>> # Explicit conversion with suggestions for better bond detection
    >>> atoms = smiles2atoms("CCO")
    >>> mol = ase2rdkit(atoms, suggestions=["CCO"])
    >>>
    >>> # Find all carbon atoms
    >>> match_substructure(mol, "[#6]")
    ((0,), (1,))
    >>>
    >>> # Find C-O bond with hydrogens included
    >>> match_substructure(mol, "CO", hydrogens="include")
    ((1, 6, 7, 2, 8),)
    >>>
    >>> # Use atom mapping to select specific atoms in order
    >>> match_substructure(mol, "[C:2][C:1]O", mapped_only=True)
    ((1, 0),)  # Returns in map order: C:2 then C:1
    >>>
    >>> # Multiple matches in propanol
    >>> propanol = ase2rdkit(smiles2atoms("CCCO"))
    >>> match_substructure(propanol, "[#6]")
    ((0,), (1,), (2,))
    """
    # Parse pattern
    patt = Chem.MolFromSmarts(smarts_or_smiles)
    if patt is None:
        patt = Chem.MolFromSmiles(smarts_or_smiles)
    if patt is None:
        raise ValueError(f"Invalid SMARTS/SMILES: {smarts_or_smiles}")

    # Extract and validate atom mappings
    mapped_pattern_indices = []
    atom_map_numbers = []
    for atom in patt.GetAtoms():
        if atom.GetAtomMapNum() > 0:
            map_num = atom.GetAtomMapNum()
            if map_num in atom_map_numbers:
                raise ValueError(f"Atom map label '{map_num}' is used multiple times")
            atom_map_numbers.append(map_num)
            mapped_pattern_indices.append(atom.GetIdx())

    # Sort mapped indices by map number to preserve ordering
    if mapped_pattern_indices:
        map_index_pairs = [
            (patt.GetAtomWithIdx(idx).GetAtomMapNum(), idx)
            for idx in mapped_pattern_indices
        ]
        map_index_pairs.sort(key=lambda x: x[0])
        mapped_pattern_indices = [idx for _, idx in map_index_pairs]

    # Perform substructure matching
    all_matches = mol.GetSubstructMatches(patt)
    if not all_matches:
        return ()

    # Process each match
    processed_matches = []
    for match in all_matches:
        # Apply mapped_only filter if requested
        if mapped_only and mapped_pattern_indices:
            core_atoms = tuple(match[idx] for idx in mapped_pattern_indices)
        else:
            core_atoms = match

        # Apply hydrogen handling
        if hydrogens == "exclude":
            # Keep only heavy atoms
            final_atoms = tuple(
                idx for idx in core_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1
            )
        elif hydrogens == "include":
            # Include heavy atoms and their bonded hydrogens
            final_atoms_list = []
            already_added = set()
            for idx in core_atoms:
                # Skip if already added (can happen with explicit [H] in pattern)
                if idx not in already_added:
                    final_atoms_list.append(idx)
                    already_added.add(idx)
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() != 1:  # Is heavy atom
                    h_neighbors = sorted(
                        neighbor.GetIdx()
                        for neighbor in atom.GetNeighbors()
                        if neighbor.GetAtomicNum() == 1
                        and neighbor.GetIdx() not in already_added
                    )
                    final_atoms_list.extend(h_neighbors)
                    already_added.update(h_neighbors)
            final_atoms = tuple(final_atoms_list)
        elif hydrogens == "isolated":
            # Return only hydrogens bonded to matched heavy atoms
            final_atoms_list = []
            for idx in core_atoms:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() != 1:  # Is heavy atom
                    h_neighbors = sorted(
                        neighbor.GetIdx()
                        for neighbor in atom.GetNeighbors()
                        if neighbor.GetAtomicNum() == 1
                    )
                    final_atoms_list.extend(h_neighbors)
            final_atoms = tuple(final_atoms_list)
        else:
            raise ValueError(
                f"Invalid value for `hydrogens`: {hydrogens!r}. "
                "Expected one of 'include', 'exclude', 'isolated'."
            )

        if final_atoms:
            processed_matches.append(final_atoms)

    return tuple(processed_matches)


def group_matches_by_fragment(
    mol: Chem.Mol, matches: tuple[tuple[int, ...], ...]
) -> list[list[int]]:
    """Group matched atom indices by disconnected molecular fragments.

    This function takes substructure matches and organizes them by which
    disconnected fragment each match belongs to. Duplicate atoms within
    each fragment are removed.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule containing one or more disconnected fragments.
    matches : tuple[tuple[int, ...], ...]
        Matches returned from ``match_substructure``, where each tuple
        contains atom indices for one match.

    Returns
    -------
    list[list[int]]
        List of atom index lists, one per fragment. Each inner list contains
        unique atom indices for all matches within that fragment. Fragments
        with no matches are omitted.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from molify import match_substructure, group_matches_by_fragment
    >>>
    >>> # Molecule with two disconnected fragments
    >>> mol = Chem.MolFromSmiles("CCO.CF")
    >>> matches = match_substructure(mol, "[#6]")
    >>> matches
    ((0,), (1,), (3,))
    >>>
    >>> # Group by fragment
    >>> group_matches_by_fragment(mol, matches)
    [[0, 1], [3]]
    """
    if not matches:
        return []

    fragment_sets = [set(frag) for frag in Chem.GetMolFrags(mol, asMols=False)]

    grouped = [[] for _ in fragment_sets]
    for match in matches:
        for frag_idx, frag_atoms in enumerate(fragment_sets):
            if set(match).issubset(frag_atoms):
                grouped[frag_idx].extend(match)
                break

    # Remove duplicates within each fragment while preserving order
    result = []
    for group in grouped:
        if group:
            # Remove duplicates while preserving order
            seen = set()
            unique = [x for x in group if not (x in seen or seen.add(x))]
            result.append(unique)

    return result


def get_substructures(
    atoms: ase.Atoms,
    pattern: str,
    **kwargs,
) -> list[ase.Atoms]:
    """Extract all matched substructures from an ASE Atoms object.

    This function converts ASE Atoms to RDKit Mol, finds all substructure
    matches, and returns ASE Atoms objects for each match.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure to search in.
    pattern : str
        SMARTS or SMILES pattern to match.
    **kwargs
        Additional keyword arguments passed to ``ase2rdkit`` for molecule
        conversion (e.g., ``suggestions`` for bond detection hints).

    Returns
    -------
    list[ase.Atoms]
        List of ASE Atoms objects, each containing one matched substructure.

    Examples
    --------
    >>> from molify import smiles2atoms, get_substructures
    >>>
    >>> propanol = smiles2atoms("CCCO")
    >>> carbons = get_substructures(propanol, "[#6]", suggestions=["CCCO"])
    >>> len(carbons)
    3
    """
    mol = ase2rdkit(atoms, **kwargs)
    matches = match_substructure(mol, pattern)
    return [atoms[list(match)] for match in matches]


def iter_fragments(atoms: ase.Atoms) -> list[ase.Atoms]:
    """Iterate over connected molecular fragments in an ASE Atoms object.

    If a 'connectivity' field is present in ``atoms.info``, it will be used
    to determine fragments. Otherwise, ``ase.build.separate`` will be used.

    Parameters
    ----------
    atoms : ase.Atoms
        A structure that may contain one or more molecular fragments.

    Yields
    ------
    ase.Atoms
        Each connected component (fragment) in the input structure.

    Examples
    --------
    >>> from molify import smiles2atoms
    >>> from rdkit.Chem import CombineMols
    >>>
    >>> # Create multi-fragment system
    >>> ethanol = smiles2atoms("CCO")
    >>> methanol = smiles2atoms("CO")
    >>> combined = ethanol + methanol
    >>>
    >>> # Iterate over fragments
    >>> fragments = list(iter_fragments(combined))
    >>> len(fragments)
    2
    """
    if GraphAttr.CONNECTIVITY in atoms.info:
        # connectivity is a list of tuples (i, j, bond_type)
        connectivity = atoms.info[GraphAttr.CONNECTIVITY]
        for component in find_connected_components(connectivity):
            yield atoms[list(component)]
    else:
        for molecule in separate(atoms):
            yield molecule


def _collect_highlighted_fragments(mol, args, alpha):
    """Helper function to collect and process fragment highlights."""
    frags = Chem.GetMolFrags(mol, asMols=True)
    frag_indices = Chem.GetMolFrags(mol, asMols=False)

    candidate_mols = []
    candidate_highlights = []
    candidate_colors = []

    # Collect all selected indices from all argument lists
    all_selected_indices = set()
    for atom_list in args:
        all_selected_indices.update(atom_list)

    # Get colors from matplotlib's tab10 colormap and add alpha
    colors = plt.cm.tab10.colors
    highlight_colors = [colors[i % len(colors)] + (alpha,) for i in range(len(args))]

    for i, frag in enumerate(frags):
        original_indices_in_frag = set(frag_indices[i])

        # Check if this fragment contains any of the selected atoms
        if not all_selected_indices.isdisjoint(original_indices_in_frag):
            candidate_mols.append(frag)

            # Map original indices to the new indices within the fragment
            original_to_frag_map = {
                orig_idx: new_idx for new_idx, orig_idx in enumerate(frag_indices[i])
            }

            current_highlights = []
            current_colors = {}

            # Process each argument list with its corresponding color
            for arg_idx, atom_list in enumerate(args):
                color = highlight_colors[arg_idx]
                for idx in atom_list:
                    if idx in original_to_frag_map:
                        frag_idx = original_to_frag_map[idx]
                        if frag_idx not in current_highlights:
                            current_highlights.append(frag_idx)
                        # Later argument lists take precedence for coloring
                        current_colors[frag_idx] = color

            candidate_highlights.append(current_highlights)
            candidate_colors.append(current_colors)

    return candidate_mols, candidate_highlights, candidate_colors


def _filter_unique_molecules(candidate_mols, candidate_highlights, candidate_colors):
    """Helper function to filter for unique molecular structures."""
    mols_to_draw = []
    highlight_lists = []
    highlight_colors = []
    seen_smiles = set()

    for i, candidate_mol in enumerate(candidate_mols):
        # Generate canonical SMILES to identify unique structures
        mol_no_hs = Chem.RemoveHs(candidate_mol)
        smi = Chem.MolToSmiles(mol_no_hs, canonical=True)

        if smi not in seen_smiles:
            seen_smiles.add(smi)
            mols_to_draw.append(candidate_mol)
            highlight_lists.append(candidate_highlights[i])
            highlight_colors.append(candidate_colors[i])

    return mols_to_draw, highlight_lists, highlight_colors


def visualize_selected_molecules(
    mol: Chem.Mol,
    *args,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (200, 200),
    legends: list[str] | None = None,
    alpha: float = 0.5,
):
    """Visualize molecules with optional atom highlighting.

    If no atom selections are provided, displays the molecule without highlights.
    Duplicate molecular structures will only be plotted once.

    Parameters
    ----------
    mol : Chem.Mol
        The RDKit molecule object, which may contain multiple fragments.
    *args : list[int] or tuple[int]
        Variable number of lists/tuples containing atom indices to be highlighted.
        Each selection will be assigned a different color from matplotlib's tab10
        colormap. If no arguments provided, displays the molecule without highlights.
    mols_per_row : int, default=4
        Number of molecules per row in the grid.
    sub_img_size : tuple[int, int], default=(200, 200)
        Size of each molecule image in pixels.
    legends : list[str], optional
        Custom legends for each molecule. If None, default legends will be used.
    alpha : float, default=0.5
        Transparency level for the highlighted atoms (0.0 = fully transparent,
        1.0 = opaque).

    Returns
    -------
    PIL.Image
        A PIL image object of the grid.

    Examples
    --------
    >>> from molify import smiles2atoms, ase2rdkit, match_substructure
    >>> from molify import visualize_selected_molecules
    >>>
    >>> # Create and convert molecule
    >>> mol = ase2rdkit(smiles2atoms("Cc1ccccc1"))
    >>>
    >>> # Find aromatic and aliphatic carbons
    >>> aromatic = match_substructure(mol, "c")
    >>> aliphatic = match_substructure(mol, "[C;!c]")
    >>>
    >>> # Flatten matches for visualization
    >>> aromatic_flat = [idx for match in aromatic for idx in match]
    >>> aliphatic_flat = [idx for match in aliphatic for idx in match]
    >>>
    >>> # Visualize with highlights
    >>> visualize_selected_molecules(mol, aromatic_flat, aliphatic_flat)
    <PIL.Image>
    """
    # Handle empty args case - display molecule without highlights
    if not args:
        img = Draw.MolsToGridImage(
            [mol],
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            legends=legends if legends is not None else ["Molecule 0"],
        )
        return img

    # Collect highlighted fragments
    candidate_mols, candidate_highlights, candidate_colors = (
        _collect_highlighted_fragments(mol, args, alpha)
    )

    if not candidate_mols:
        print("No molecules to draw with the given selections.")
        return None

    # Filter for unique molecules
    mols_to_draw, highlight_lists, highlight_colors = _filter_unique_molecules(
        candidate_mols, candidate_highlights, candidate_colors
    )

    # Draw the grid
    final_legends = (
        legends
        if legends is not None
        else [f"Molecule {i}" for i in range(len(mols_to_draw))]
    )

    img = Draw.MolsToGridImage(
        mols_to_draw,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=final_legends,
        highlightAtomLists=highlight_lists,
        highlightAtomColors=highlight_colors,
    )
    return img
