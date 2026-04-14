"""String constants used as attribute keys across molify conversions."""

from enum import StrEnum


class NodeAttr(StrEnum):
    """Attribute keys stored on networkx graph nodes."""

    POSITION = "position"
    ATOMIC_NUMBER = "atomic_number"
    ORIGINAL_INDEX = "original_index"
    CHARGE = "charge"


class EdgeAttr(StrEnum):
    """Attribute keys stored on networkx graph edges."""

    BOND_ORDER = "bond_order"


class GraphAttr(StrEnum):
    """Attribute keys stored on networkx graph-level or ASE atoms.info dicts."""

    PBC = "pbc"
    CELL = "cell"
    CONNECTIVITY = "connectivity"
    SMILES = "smiles"
