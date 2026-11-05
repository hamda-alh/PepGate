import ast
from pathlib import Path

import pandas as pd


ALLOWED_STATES = {
    "none",
    "D",
    "NMe",
    "D+NMe",
}


def load_monomer_table(path: str | Path) -> pd.DataFrame:
    """
    Load the CycPeptMPDB monomer table.
    """
    return pd.read_csv(path, low_memory=False)


def parse_iupac_backbone(iupac_condensed: str) -> tuple[str | None, str]:
    """
    Separate side-chain identity from the PepGate backbone state.

    Examples
    --------
    H-Ala-OH       -> ("Ala", "none")
    H-D-Ala-OH     -> ("Ala", "D")
    Me-Ala-OH      -> ("Ala", "NMe")
    Me-D-Ala-OH    -> ("Ala", "D+NMe")

    Critically, side-chain substitutions are preserved:

    H-Ala(tBu)-OH     -> ("Ala(tBu)", "none")
    H-D-Ala(tBu)-OH   -> ("Ala(tBu)", "D")
    """

    if pd.isna(iupac_condensed):
        return None, "UNSUPPORTED"

    value = str(iupac_condensed).strip()

    if value == "N.D":
        return None, "UNSUPPORTED"

    # Determine backbone N-substitution.
    if value.startswith("H-"):
        remainder = value[2:]
        n_methylated = False

    elif value.startswith("Me-"):
        remainder = value[3:]
        n_methylated = True

    else:
        # Some residues have other N-substitutions.
        # These are outside PepGate's current edit space.
        return None, "OTHER_N_SUB"

    # Determine L/D stereochemistry.
    is_d = False

    if remainder.startswith("D-"):
        remainder = remainder[2:]
        is_d = True

    # Remove the common terminal group.
    if remainder.endswith("-OH"):
        remainder = remainder[:-3]

    if n_methylated and is_d:
        state = "D+NMe"
    elif n_methylated:
        state = "NMe"
    elif is_d:
        state = "D"
    else:
        state = "none"

    return remainder, state


def build_monomer_map(
    monomer_df: pd.DataFrame,
) -> dict[str, dict[str, str | None]]:
    """
    Build a lookup table from CycPeptMPDB monomer symbol to:

    - side-chain identity
    - backbone modification state
    """

    monomer_map = {}

    for _, row in monomer_df.iterrows():
        symbol = str(row["Symbol"])

        sidechain, state = parse_iupac_backbone(
            row["IUPAC_Condensed"]
        )

        monomer_map[symbol] = {
            "sidechain": sidechain,
            "state": state,
        }

    return monomer_map


def parse_sequence(sequence_string: str) -> list[str]:
    """
    Parse the CycPeptMPDB Sequence column.
    """

    if pd.isna(sequence_string):
        return []

    sequence = ast.literal_eval(sequence_string)

    if not isinstance(sequence, list):
        raise ValueError(
            f"Expected list, got {type(sequence)}"
        )

    return [str(x) for x in sequence]


def annotate_sequence(
    sequence_string: str,
    monomer_map: dict,
) -> dict:
    """
    Convert a peptide into:

    1. exact side-chain identities
    2. PepGate backbone modification pattern
    """

    symbols = parse_sequence(sequence_string)

    sidechains = []
    backbone = []

    for symbol in symbols:
        info = monomer_map.get(symbol)

        if info is None:
            return {
                "supported": False,
                "reason": f"Unknown monomer: {symbol}",
                "sidechains": None,
                "backbone": None,
            }

        state = info["state"]
        sidechain = info["sidechain"]

        if state not in ALLOWED_STATES:
            return {
                "supported": False,
                "reason": f"Unsupported backbone type: {symbol}",
                "sidechains": None,
                "backbone": None,
            }

        sidechains.append(sidechain)
        backbone.append(state)

    return {
        "supported": True,
        "reason": None,
        "sidechains": tuple(sidechains),
        "backbone": tuple(backbone),
    }

def canonicalize_ring(
    sidechains: tuple[str, ...],
    backbone: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Canonicalize a cyclic peptide while keeping the backbone
    modification pattern aligned with the side chains.
    """

    if len(sidechains) != len(backbone):
        raise ValueError(
            "Side-chain and backbone sequences must have the same length."
        )

    if len(sidechains) == 0:
        return tuple(), tuple()

    rotations = []

    for i in range(len(sidechains)):
        rotated_sidechains = sidechains[i:] + sidechains[:i]
        rotated_backbone = backbone[i:] + backbone[:i]

        rotations.append(
            (
                rotated_sidechains,
                rotated_backbone,
            )
        )

    minimum_sidechain = min(
        sidechain
        for sidechain, backbone_state in rotations
    )

    candidates = [
        pair
        for pair in rotations
        if pair[0] == minimum_sidechain
    ]

    return min(
        candidates,
        key=lambda pair: pair[1],
    )