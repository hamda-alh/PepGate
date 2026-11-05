import statistics

from optimizer import (
    exhaustive_search,
    generate_and_rank,
    guided_search,
    random_search,
)


parent = (
    "none",
    "D",
    "none",
    "none",
)


def dummy_score(pattern):
    score = 0.0

    for state in pattern:
        if state == "NMe":
            score += 1.0

        elif state == "D":
            score += 0.25

        elif state == "D+NMe":
            score += 1.25

    return score


# --------------------------------------------------
# True optimum
# --------------------------------------------------

exhaustive = exhaustive_search(
    parent=parent,
    score_fn=dummy_score,
    budget=2,
)

optimal_pattern, optimal_score = exhaustive[0]


# --------------------------------------------------
# Repeated benchmark
# --------------------------------------------------

N_SEEDS = 100

random_regrets = []
rank_regrets = []
guided_regrets = []

random_scores = []
rank_scores = []
guided_scores = []

random_success = 0
rank_success = 0
guided_success = 0


for seed in range(N_SEEDS):

    # -------------------------------
    # Random search
    # -------------------------------

    random_results = random_search(
        parent=parent,
        score_fn=dummy_score,
        budget=2,
        n_samples=19,
        elementary_only=False,
        seed=seed,
    )

    _, random_score = random_results[0]

    random_scores.append(random_score)

    random_regret = (
        optimal_score - random_score
    )

    random_regrets.append(random_regret)

    if random_regret == 0:
        random_success += 1


    # -------------------------------
    # Generate-and-rank
    # -------------------------------

    rank_results = generate_and_rank(
        parent=parent,
        score_fn=dummy_score,
        budget=2,
        n_candidates=19,
        seed=seed,
    )

    _, rank_score = rank_results[0]

    rank_scores.append(rank_score)

    rank_regret = (
        optimal_score - rank_score
    )

    rank_regrets.append(rank_regret)

    if rank_regret == 0:
        rank_success += 1


    # -------------------------------
    # Guided search
    # -------------------------------

    guided = guided_search(
        parent=parent,
        score_fn=dummy_score,
        budget=2,
        beam_width=3,
        max_oracle_calls=20,
        max_rounds=5,
        elementary_only=False,
        seed=seed,
    )

    guided_score = guided["best_score"]

    guided_scores.append(guided_score)

    guided_regret = (
        optimal_score - guided_score
    )

    guided_regrets.append(
        guided_regret
    )

    if guided_regret == 0:
        guided_success += 1


# --------------------------------------------------
# Summary function
# --------------------------------------------------

def summarize(
    name,
    scores,
    regrets,
    successes,
):

    print()
    print(name)

    print(
        "  Mean best score:",
        round(statistics.mean(scores), 3),
    )

    print(
        "  Mean regret:",
        round(statistics.mean(regrets), 3),
    )

    print(
        "  Median regret:",
        round(statistics.median(regrets), 3),
    )

    print(
        "  Regret SD:",
        round(statistics.stdev(regrets), 3),
    )

    print(
        "  Optimum found:",
        f"{successes}/{N_SEEDS}",
        f"({100 * successes / N_SEEDS:.1f}%)",
    )


print("=== Repeated PepGate search benchmark ===")

print()
print(
    "True optimum:",
    optimal_pattern,
    "score =",
    optimal_score,
)

summarize(
    "Random search",
    random_scores,
    random_regrets,
    random_success,
)

summarize(
    "Generate-and-rank",
    rank_scores,
    rank_regrets,
    rank_success,
)

summarize(
    "Guided search",
    guided_scores,
    guided_regrets,
    guided_success,
)