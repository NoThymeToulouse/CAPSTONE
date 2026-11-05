
import numpy as np

from itertools import combinations
from collections import defaultdict

from sklearn.preprocessing import KBinsDiscretizer


# ======================================
# ROUGH SET DEPENDENCY CALCULATION
# ======================================

def rough_dependency(X_discrete, y, columns):

    groups = defaultdict(list)

    for row, label in zip(X_discrete, y):

        key = tuple(row[list(columns)])

        groups[key].append(label)

    positive_region = 0

    for labels in groups.values():

        # A group belongs to the positive
        # region if all labels are identical.

        if len(set(labels)) == 1:

            positive_region += len(labels)

    dependency = positive_region / len(y)

    return dependency


# ======================================
# ROUGH SET FEATURE REDUCTION
# ======================================

def rough_reduct(X, y, n_bins=3):

    X = np.asarray(X)
    y = np.asarray(y)

    # Discretise continuous features

    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="quantile"
    )

    X_discrete = discretizer.fit_transform(X)

    X_discrete = X_discrete.astype(int)

    n_features = X.shape[1]

    all_features = tuple(range(n_features))

    # Measure full dependency

    full_dependency = rough_dependency(
        X_discrete,
        y,
        all_features
    )

    print(
        "Full rough dependency:",
        round(full_dependency, 4)
    )

    # If no positive region, feature reduction inconclusive.

    if full_dependency == 0:

        print("No informative exact reduct found.")

        return list(all_features)

    # minimal reduct

    for subset_size in range(1, n_features + 1):

        for subset in combinations(
            all_features,
            subset_size
        ):

            dependency = rough_dependency(
                X_discrete,
                y,
                subset
            )

            if dependency >= full_dependency - 1e-12:

                print(
                    "Selected feature indices:",
                    subset
                )

                print(
                    "Selected dependency:",
                    round(dependency, 4)
                )

                return list(subset)

    # Defensive fallback

    return list(all_features)
