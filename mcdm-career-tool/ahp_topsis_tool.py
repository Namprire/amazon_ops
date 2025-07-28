import numpy as np
from pymcdm.methods import TOPSIS
from pymcdm.helpers import rrankdata
from pymcdm.weights import AHP


def build_ahp_weights(pairwise_matrix):
    """
    Compute AHP weights using a pairwise comparison matrix.
    """
    model = AHP()
    weights = model(pairwise_matrix)
    return weights


def apply_time_discounting(weights, discount_rate):
    """
    Apply exponential time discounting to criteria weights.
    """
    decay = np.exp(-discount_rate * np.arange(len(weights)))
    discounted = weights * decay
    return discounted / np.sum(discounted)


def rank_with_topsis(matrix, weights, types):
    """
    Rank career options using TOPSIS method.
    """
    model = TOPSIS()
    scores = model(matrix, weights, types)
    ranking = rrankdata(scores)
    return ranking, scores


if __name__ == "__main__":
    # Example: AHP matrix (Income, Work Hours, Stability)
    ahp_matrix = np.array([
        [1,   3,   0.5],
        [1/3, 1,   1/7],
        [2,   7,   1]
    ])

    # Career options:
    # [Income (↑), Work Hours (↓), Stability (↑)]
    decision_matrix = np.array([
        [9, 60, 4],  # High-Risk High-Reward
        [7, 45, 6],  # Balanced Private
        [5, 40, 9],  # Public Sector
        [3, 30, 5],  # Low-Income Flexible
        [4, 55, 7]   # Mission-Driven
    ])

    # 1 = benefit, -1 = cost
    types = np.array([1, -1, 1])

    # Step 1: AHP Weights
    raw_weights = build_ahp_weights(ahp_matrix)

    # Step 2: Apply time discounting (e.g., 0.2 = moderate short-term bias)
    discounted_weights = apply_time_discounting(raw_weights, discount_rate=0.2)

    # Step 3: Rank with TOPSIS
    ranks, scores = rank_with_topsis(decision_matrix, discounted_weights, types)

    # Output
    print("Career Option Rankings:\n")
    for i, (rank, score) in enumerate(zip(ranks, scores)):
        print(f"Option {i+1}: Rank = {int(rank)}, Score = {score:.4f}")
