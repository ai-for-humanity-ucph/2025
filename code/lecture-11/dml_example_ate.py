# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "doubleml",
# ]
# ///

"""
Examples from the lecture.

Usage:
    uv run code/lecture-11/dml_example_ate.py
"""

import warnings

import doubleml as dml
from doubleml.datasets import make_irm_data
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore", category=FutureWarning)


# DGP
data = make_irm_data(theta=0.5, n_obs=1_000, dim_x=10, return_type="DataFrame")

# Inspect data
print("Data:")
print(data)

# Nuisance estimators

# # E[Y | X, D]
ml_g = RandomForestRegressor(
    n_estimators=100,
    max_features=10,
    max_depth=5,
    min_samples_leaf=2,
)
# # P(D = 1 | X)
ml_m = RandomForestClassifier(
    n_estimators=100,
    max_features=10,
    max_depth=5,
    min_samples_leaf=2,
)

# Fit DML
obj_dml_data = dml.DoubleMLData(data, "y", "d")
dml_irm_obj = dml.DoubleMLIRM(obj_dml_data, ml_g, ml_m)

dml_irm_obj.fit()

# Results
print(dml_irm_obj)
