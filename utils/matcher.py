# utils/matcher.py
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np


def match_players(broadcast_features, tacticam_features):
    # feature shape: [N, 2048]
    cost_matrix = cdist(tacticam_features, broadcast_features, metric='cosine')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 0.5:  # threshold
            matches.append((r, c))  # tacticam[r] → broadcast[c]
    return matches
