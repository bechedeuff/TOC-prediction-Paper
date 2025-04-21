"""
Main script to run the TOC prediction training pipeline.
"""
import os
import sys

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.xgb import XGBModel
from src.models.gbdt import GBDTModel
from src.models.mlp import MLPModel
from src.training.train import Trainer
from src.utils.visualization import Visualization

from src.config.configurations import (
    EACH_MODEL_PATH,
    HPT_TUNING_PATH,
    RMSE_RESULTS_PATH,
    FIGS_PATH,
    XGB_PBOUNDS,
    GBDT_PBOUNDS,
    MLP_PBOUNDS,
    MODEL_NAMES
)

def create_directories():
    """Create dirs if they don't exist."""
    directories = [
        EACH_MODEL_PATH,
        HPT_TUNING_PATH,
        RMSE_RESULTS_PATH,
        FIGS_PATH,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    create_directories()
    
    # Select models to train with their respective parameter bounds
    trainers = [
        (XGBModel, XGB_PBOUNDS),
        (GBDTModel, GBDT_PBOUNDS),
        (MLPModel, MLP_PBOUNDS),
    ]
    
    # Train each model
    results = {}
    for model_class, pbounds in trainers:
        print(f"\nTraining {model_class.__name__}...")
        trainer = Trainer(model_class, pbounds)
        results[model_class.__name__] = trainer.train()
    
    # Plot comparison between all models
    Visualization.plot_models_comparison(MODEL_NAMES)
    
    return results

if __name__ == "__main__":
    main() 