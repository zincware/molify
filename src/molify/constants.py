"""String constants used as attribute keys across molify conversions."""

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of StrEnum for Python 3.10."""


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
