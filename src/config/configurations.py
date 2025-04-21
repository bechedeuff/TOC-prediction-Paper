"""
Configuration file for TOC prediction project.
Contains all configurable parameters used across the project.
"""
import os

# Root dir
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model training settings
EXPERIMENT_NAME = "experiment_1"
N_ITER = 500
N_SPLITS = 10

# Data and results paths
REPORTS_PATH = os.path.join(PROJECT_ROOT, "results", EXPERIMENT_NAME)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "df_5_wells_fe.pkl")

# Results specific paths
EACH_MODEL_PATH = os.path.join(REPORTS_PATH, "each_model")
HPT_TUNING_PATH = os.path.join(REPORTS_PATH, "hpt_tuning")
RMSE_RESULTS_PATH = os.path.join(REPORTS_PATH, "rmse_results")
FIGS_PATH = os.path.join(REPORTS_PATH, "figs")

# Training data
FEATURE_COLUMNS = ["LAT", "LON", "WELL_ID", "GR", "RHOB", "DT", "RT", "NPHI"]
WELLS = None # Or all wells to train on ["WELL1", "WELL2", "WELL3"]
MODEL_NAMES = ["XGB", "GBDT", "MLP"]

# Plot settings
PLOT_COLS = ["GR", "RHOB", "DT", "RT", "NPHI"]
PLOT_FIGSIZE = (15, 5)
PLOT_DPI = 300

# Bayesian Optimization settings
BAYES_KAPPA = 2.5
BAYES_XI = 0.0
BAYES_INIT_POINTS = 5 

# Algorithms settings
XGB_PBOUNDS = {
    "n_estimators": (50, 1000),
    "max_depth": (1, 10),
    "min_child_weight": (1, 20),
    "eta": (0.001, 0.05),
    "subsample": (0.5, 1),
    "colsample_bytree": (0.5, 1),
    "reg_alpha": (0, 10),
    "reg_lambda": (1, 10),
}

GBDT_PBOUNDS = {
    "n_estimators": (50, 1000),
    "max_depth": (1, 10),
    "min_samples_split": (10, 30),
    "min_samples_leaf": (10, 30),
    "max_features": (0.1, 0.8),
    "learning_rate": (0.001, 0.05),
    "tol": (0.0001, 0.1),
}

MLP_PBOUNDS = {
    "hidden_layer_sizes": (0, 2),
    "activation": (0, 1),
    "learning_rate": (0, 2),
    "max_iter": (500, 2000),
    "alpha": (0.001, 1),
    "tol": (0.0001, 0.01),
}
