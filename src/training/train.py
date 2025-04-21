"""
Main training module that orchestrates the training process.
"""
import os
import pandas as pd
import json
from datetime import datetime

from src.config.configurations import (
    RMSE_RESULTS_PATH,
    DATA_PATH,
    FEATURE_COLUMNS,
    WELLS,
    N_SPLITS,
    N_ITER,
    EXPERIMENT_NAME,
    BAYES_KAPPA,
    BAYES_XI,
    BAYES_INIT_POINTS,
    REPORTS_PATH,
)

from src.utils.data_loader import DataLoader
from src.utils.metrics import Metrics
from src.utils.visualization import Visualization
from src.training.hyperparameter_tuning import HyperparameterTuning

class Trainer:
    def __init__(self, model_class, pbounds):
        self.model_class = model_class
        self.pbounds = pbounds
        self.model = model_class()
        self.data_loader = DataLoader()
        self.metrics = Metrics()
        self.visualization = Visualization()
        self.hpt = HyperparameterTuning(model_class, pbounds)
        
    def train(self):
        """Main training pipeline."""
        # Load and prepare data
        df = self.data_loader.load_data(DATA_PATH)
        df = self.data_loader.filter_wells(df, WELLS)
        
        # Prepare features and target
        X, y, target_bins = self.model.prepare_data(df, FEATURE_COLUMNS)
        
        # Store target_bins back in df for later use
        df["target_bins"] = target_bins
        
        # Perform hyperparameter tuning
        hpt_results = self.hpt.tune(X, y, target_bins)
        
        # Get best parameters
        best_params = hpt_results.iloc[0].to_dict()
        best_params.pop("rmse")
        best_params.pop("model")
        
        # Create and evaluate model with best parameters
        model = self.model.create_model(**best_params)
        cv_results = self.model.evaluate_model(model, X, y, df)
        
        # Calculate and save metrics
        metrics = self.metrics.calculate_metrics(y, cv_results["all_predictions"])
        residuals = self.metrics.calculate_residuals(y, cv_results["all_predictions"])
        error_range = self.metrics.calculate_error_range(residuals["percentual"])
        
        # Save results
        results = {
            "model": self.model.name,
            "metrics": metrics,
            "residuals": residuals,
            "error_range": error_range,
            "best_params": best_params,
        }
        
        pd.DataFrame([results]).to_csv(
            os.path.join(RMSE_RESULTS_PATH, f"{self.model.name}_final_results.csv"),
            index=False,
        )
        
        self.visualization.plot_learning_curve(
            model,
            X,
            y,
            target_bins,
            self.model.name,
            self.model.skf
        )
        
        self.visualization.plot_prediction_intervals(
            df,
            y,
            cv_results["all_predictions"],
            self.model.name,
        )
        
        self.visualization.plot_prediction_per_well(
            df, y, cv_results["all_predictions"], self.model.name
        )
        
        self.visualization.plot_ml_comparison_passey(
            df, y, cv_results["all_predictions"], self.model.name, metrics
        )
        
        # Save metadata
        self._save_experiment_metadata(df, best_params, metrics)
        
        return results 
    
    def _save_experiment_metadata(self, df, best_params, metrics):
        """Save experiment metadata to JSON."""
        
        metadata_path = os.path.join(REPORTS_PATH, f"metadata_{EXPERIMENT_NAME}.json")
        
        # Load existing metadata if it exists
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "experiment_name": EXPERIMENT_NAME,
                "timestamp": datetime.now().isoformat(),
                "models": {},
                "training": {
                    "n_splits": N_SPLITS,
                    "n_iterations": N_ITER,
                    "feature_columns": FEATURE_COLUMNS,
                    "wells": WELLS if WELLS else "all",
                    "target_column": "TOC",
                    "data_statistics": {
                        "data_points": len(df),
                        "toc_min": float(df["TOC"].min()),
                        "toc_max": float(df["TOC"].max()),
                        "toc_mean": float(df["TOC"].mean()),
                        "toc_std": float(df["TOC"].std()),
                        "wells_count": len(df["WELLNAME"].unique()),
                        "wells_names": df["WELLNAME"].unique().tolist()
                    }
                },
                "bayesian_optimization": {
                    "kappa": BAYES_KAPPA,
                    "xi": BAYES_XI,
                    "init_points": BAYES_INIT_POINTS,
                    "utility_function": "ucb"
                },
                "performance": {}
            }
        
        # Update model-specific information
        metadata["models"][self.model.name] = {
            "best_parameters": best_params
        }
        
        # Update performance metrics for this model
        metadata["performance"][self.model.name] = {
            "metrics": metrics
        }
        
        # Save updated metadata
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)