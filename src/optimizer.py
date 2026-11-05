from itertools import product


BACKBONE_STATES = (
    "none",
    "D",
    "NMe",
    "D+NMe",
)


STATE_FEATURES = {
    "none": frozenset(),
    "D": frozenset({"D"}),
    "NMe": frozenset({"NMe"}),
    "D+NMe": frozenset({"D", "NMe"}),
}


def hamming_distance(
    pattern_a: tuple[str, ...],
    pattern_b: tuple[str, ...],
) -> int:
    """
    Count how many backbone positions differ.
    """

    if len(pattern_a) != len(pattern_b):
        raise ValueError(
            "Backbone patterns must have the same length."
        )

    return sum(
        a != b
        for a, b in zip(pattern_a, pattern_b)
    )


def within_edit_budget(
    candidate: tuple[str, ...],
    parent: tuple[str, ...],
    budget: int,
) -> bool:
    """
    Return True when the candidate differs from the parent
    at no more than `budget` positions.
    """

    return (
        hamming_distance(candidate, parent)
        <= budget
    )


def is_elementary_transition(
    state_a: str,
    state_b: str,
) -> bool:
    """
    Return True if the transition adds or removes exactly
    one elementary backbone feature: D or NMe.

    Examples
    --------
    none <-> D        True
    none <-> NMe      True
    D <-> D+NMe       True
    NMe <-> D+NMe     True

    none <-> D+NMe    False
    D <-> NMe         False
    """

    features_a = STATE_FEATURES[state_a]
    features_b = STATE_FEATURES[state_b]

    difference = (
        features_a.symmetric_difference(
            features_b
        )
    )

    return len(difference) == 1


def generate_neighbors(
    pattern: tuple[str, ...],
    parent: tuple[str, ...],
    budget: int,
    editable_positions: set[int] | None = None,
    elementary_only: bool = False,
) -> list[tuple[str, ...]]:
    """
    Generate backbone patterns differing from `pattern`
    at exactly one position.

    Candidates that exceed the Hamming edit budget relative
    to the original parent are removed.
    """

    if len(pattern) != len(parent):
        raise ValueError(
            "Pattern and parent must have the same length."
        )

    if editable_positions is None:
        editable_positions = set(
            range(len(pattern))
        )

    neighbors = []

    for position in sorted(editable_positions):

        current_state = pattern[position]

        for new_state in BACKBONE_STATES:

            if new_state == current_state:
                continue

            if (
                elementary_only
                and not is_elementary_transition(
                    current_state,
                    new_state,
                )
            ):
                continue

            candidate = list(pattern)
            candidate[position] = new_state
            candidate = tuple(candidate)

            if within_edit_budget(
                candidate,
                parent,
                budget,
            ):
                neighbors.append(candidate)

    return list(dict.fromkeys(neighbors))

import random
from collections.abc import Callable


def random_search(
    parent: tuple[str, ...],
    score_fn: Callable[[tuple[str, ...]], float],
    budget: int,
    n_samples: int = 100,
    editable_positions: set[int] | None = None,
    elementary_only: bool = False,
    seed: int = 42,
) -> list[tuple[tuple[str, ...], float]]:
    """
    Randomly explore legal backbone patterns around a parent.

    Parameters
    ----------
    parent:
        Original backbone pattern.

    score_fn:
        Function that accepts a backbone pattern and returns
        a numerical score. Later this will be the permeability
        oracle.

    budget:
        Maximum Hamming distance from the original parent.

    n_samples:
        Maximum number of unique candidates to evaluate.

    editable_positions:
        Positions allowed to change. If None, all positions
        are editable.

    elementary_only:
        Restrict each search transition to addition/removal of
        one elementary D or NMe feature.

    seed:
        Random seed for reproducibility.

    Returns
    -------
    Ranked list of:
        (candidate_pattern, score)

    highest score first.
    """

    rng = random.Random(seed)

    if editable_positions is None:
        editable_positions = set(
            range(len(parent))
        )

    visited = {parent}
    frontier = [parent]

    candidates = []

    while (
        frontier
        and len(candidates) < n_samples
    ):

        current = rng.choice(frontier)

        neighbors = generate_neighbors(
            pattern=current,
            parent=parent,
            budget=budget,
            editable_positions=editable_positions,
            elementary_only=elementary_only,
        )

        unseen = [
            candidate
            for candidate in neighbors
            if candidate not in visited
        ]

        if not unseen:
            frontier.remove(current)
            continue

        candidate = rng.choice(unseen)

        visited.add(candidate)
        frontier.append(candidate)

        score = score_fn(candidate)

        candidates.append(
            (
                candidate,
                score,
            )
        )

    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return candidates

def exhaustive_search(
    parent: tuple[str, ...],
    score_fn: Callable[[tuple[str, ...]], float],
    budget: int,
    editable_positions: set[int] | None = None,
) -> list[tuple[tuple[str, ...], float]]:
    """
    Enumerate every legal backbone pattern within the
    Hamming edit budget and rank by score.
    """

    if editable_positions is None:
        editable_positions = set(range(len(parent)))

    choices = []

    for i, state in enumerate(parent):
        if i in editable_positions:
            choices.append(BACKBONE_STATES)
        else:
            choices.append((state,))

    results = []

    for candidate in product(*choices):
        candidate = tuple(candidate)

        if not within_edit_budget(
            candidate,
            parent,
            budget,
        ):
            continue

        score = score_fn(candidate)

        results.append(
            (
                candidate,
                score,
            )
        )

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return results

def generate_and_rank(
    parent: tuple[str, ...],
    score_fn: Callable[[tuple[str, ...]], float],
    budget: int,
    n_candidates: int = 20,
    editable_positions: set[int] | None = None,
    seed: int = 42,
) -> list[tuple[tuple[str, ...], float]]:
    """
    Generate a fixed batch of legal backbone patterns,
    score them once, and rank them.

    This is a non-iterative generate-and-rank baseline.
    """

    rng = random.Random(seed)

    if editable_positions is None:
        editable_positions = set(
            range(len(parent))
        )

    # Get all legal candidates.
    all_candidates = exhaustive_search(
        parent=parent,
        score_fn=lambda pattern: 0.0,
        budget=budget,
        editable_positions=editable_positions,
    )

    # Keep only the patterns.
    patterns = [
        pattern
        for pattern, _ in all_candidates
    ]

    # Remove the unchanged parent.
    patterns = [
        pattern
        for pattern in patterns
        if pattern != parent
    ]

    # Sample without replacement.
    if n_candidates < len(patterns):
        patterns = rng.sample(
            patterns,
            n_candidates,
        )

    # Score the generated batch.
    results = [
        (
            pattern,
            score_fn(pattern),
        )
        for pattern in patterns
    ]

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return results

def guided_search(
    parent: tuple[str, ...],
    score_fn: Callable[[tuple[str, ...]], float],
    budget: int,
    beam_width: int = 3,
    max_oracle_calls: int = 20,
    max_rounds: int = 5,
    editable_positions: set[int] | None = None,
    elementary_only: bool = False,
    seed: int = 42,
):
    """
    Iterative guided optimization over backbone patterns.

    At each round:
        1. Generate legal neighbors of the current best patterns.
        2. Score unseen candidates.
        3. Keep the top-scoring candidates as the next beam.
        4. Repeat until the oracle-call or round budget is exhausted.

    Later, HELM-BERT proposals can replace the local-neighbor
    proposal step without changing the rest of this optimizer.
    """

    rng = random.Random(seed)

    if editable_positions is None:
        editable_positions = set(
            range(len(parent))
        )

    # Cache scores so we never waste an oracle call
    # evaluating the same pattern twice.
    score_cache = {}

    def evaluate(pattern):
        if pattern not in score_cache:
            score_cache[pattern] = score_fn(pattern)

        return score_cache[pattern]

    # Score the starting parent.
    parent_score = evaluate(parent)

    oracle_calls = 1

    # Beam contains the patterns we will expand next.
    beam = [parent]

    history = [
        {
            "round": 0,
            "best_pattern": parent,
            "best_score": parent_score,
            "oracle_calls": oracle_calls,
        }
    ]

    for round_number in range(
        1,
        max_rounds + 1,
    ):

        proposed = set()

        # -----------------------------------------
        # Generate candidates around current beam
        # -----------------------------------------

        for pattern in beam:

            neighbors = generate_neighbors(
                pattern=pattern,
                parent=parent,
                budget=budget,
                editable_positions=editable_positions,
                elementary_only=elementary_only,
            )

            for candidate in neighbors:

                if candidate not in score_cache:
                    proposed.add(candidate)

        # Nothing new can be explored.
        if not proposed:
            break

        proposed = sorted(proposed)

        # -----------------------------------------
        # Respect the oracle-call budget
        # -----------------------------------------

        remaining_calls = (
            max_oracle_calls
            - oracle_calls
        )

        if remaining_calls <= 0:
            break

        if len(proposed) > remaining_calls:
            proposed = rng.sample(
                proposed,
                remaining_calls,
            )

        # -----------------------------------------
        # Score proposed candidates
        # -----------------------------------------

        for candidate in proposed:
            evaluate(candidate)

        oracle_calls += len(proposed)

        # -----------------------------------------
        # Rank every pattern discovered so far
        # -----------------------------------------

        ranked = sorted(
            score_cache.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        # Keep only the strongest patterns for
        # the next optimization round.
        beam = [
            pattern
            for pattern, score in ranked[
                :beam_width
            ]
        ]

        best_pattern, best_score = ranked[0]

        history.append(
            {
                "round": round_number,
                "best_pattern": best_pattern,
                "best_score": best_score,
                "oracle_calls": oracle_calls,
            }
        )

        if oracle_calls >= max_oracle_calls:
            break

    final_results = sorted(
            score_cache.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

    best_pattern, best_score = (
        final_results[0]
    )

    return {
        "results": final_results,
        "best_pattern": best_pattern,
        "best_score": best_score,
        "parent_score": parent_score,
        "oracle_calls": oracle_calls,
        "history": history,
    }