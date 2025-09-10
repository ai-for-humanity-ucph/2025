# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scikit-learn",
# ]
# ///
"""
Examples from the lecture.

Usage:
    uv run scripts/decision_tree_example.py
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
    plot_tree,
)

iris = load_iris()
feature_names = iris.feature_names
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=2025, train_size=0.8
)


def main():
    """trees"""

    clf = DecisionTreeClassifier(
        min_samples_leaf=5,
        criterion="gini",
        random_state=2025,
    )
    clf.fit(X_train, y_train)
    print(np.mean(clf.predict(X_test) == y_test))

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    plot_tree(
        clf,
        filled=False,
        proportion=False,
        ax=axes,
        impurity=False,
        feature_names=feature_names,
    )
    fig.savefig("iris_dt.png", bbox_inches="tight", pad_inches=0)
    print("Saved decision tree to iris_dt.png")

    print()
    print("Decision tree in text:")
    print(export_text(clf, feature_names=feature_names))

    # From slides
    print("Example prediction:")
    mask = (X_test[:, 2] > 2.5) & (X_test[:, 3] < 1.5)
    i = np.where(mask)[0][0]
    print(", ".join([str(x) for x in X_test[i].tolist()]))
    print(clf.predict(X_test[i].reshape(1, 4)))
    print(y_test[i])

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    plot_tree(
        clf,
        filled=False,
        proportion=False,
        ax=axes,
        impurity=True,
        feature_names=feature_names,
    )
    plt.savefig("figs/iris_dt2.png", bbox_inches="tight", pad_inches=0)

    def gini_from_counts(
        N_j_t: np.ndarray,
        N_t: int,
    ):
        p_j_t = N_j_t / N_t
        return float(np.sum(p_j_t * (1 - p_j_t)))

    print("\n" + r"(p(1 \mid t),  p(2 \mid t),  p(3 \mid t))^{T}:")
    print(np.round(np.array([0, 45, 39]) / 84, 2))
    print("\np_L and p_R:")
    print(out := np.round(np.array([49, 35]) / 84, 2))
    (p_L, p_R) = out

    # Gini quiz
    print("\nimpurity measures:")
    print(
        i_t := round(gini_from_counts(np.array([0, 45, 39]), N_t=84), 3),
        i_t_L := round(gini_from_counts(np.array([0, 45, 4]), N_t=49), 3),
        i_t_R := round(gini_from_counts(np.array([0, 0, 35]), N_t=35), 3),
        delta_i_s_t := round(i_t - p_L * i_t_L - p_R * i_t_R, 3),
        sep="\n",
    )


if __name__ == "__main__":
    main()
