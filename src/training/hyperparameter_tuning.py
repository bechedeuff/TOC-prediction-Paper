"""
Hyperparameter tuning using Bayesian Optimization.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from src.config.configurations import N_SPLITS, N_ITER, HPT_TUNING_PATH, EACH_MODEL_PATH, BAYES_KAPPA, BAYES_XI, BAYES_INIT_POINTS, PLOT_DPI, FIGS_PATH

class HyperparameterTuning:
    def __init__(self, model_class, pbounds):
        self.model_class = model_class
        self.pbounds = pbounds
        self.model = model_class()
        
    def tune(self, X, y, target_bins):
        """Perform hyperparameter tuning using Bayesian Optimization."""
        def evaluate(**params):
            model = self.model.create_model(**params)
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
            scores = cross_validate(
                model,
                X,
                y,
                cv=cv.split(X, target_bins),  # Use target_bins for stratification
                scoring="neg_root_mean_squared_error"
            )
            return scores["test_score"].mean()
        
        bayes_util = UtilityFunction(kind="ucb", kappa=BAYES_KAPPA, xi=BAYES_XI)
        bo = BayesianOptimization(
            f=evaluate,
            pbounds=self.pbounds,
            random_state=42,
            verbose=1,
            allow_duplicate_points=True,
        )
        
        bo.maximize(
            init_points=BAYES_INIT_POINTS,
            n_iter=N_ITER,
            acquisition_function=bayes_util
        )
        
        # Plot optimization progress
        plt.figure(figsize=(5, 5))
        rmse_values = (-1 * pd.DataFrame(bo.res)["target"])
        cummin_rmse = rmse_values.cummin()
        plt.plot(cummin_rmse)
        plt.xlabel("Iterations")
        plt.ylabel("RMSE")
        plt.title(f"{self.model.name} - Bayesian Opt - RMSE - SKF {N_SPLITS} folds")
        plt.tight_layout()
        plt.savefig(
            f"{HPT_TUNING_PATH}/{self.model.name}_bayesopt_{N_ITER}-iters_{N_SPLITS}-skf.png",
            dpi=PLOT_DPI
        )
        #plt.show()
        plt.close()
        
        # Save results
        results = pd.DataFrame(bo.res)
        results_params = results["params"].apply(pd.Series)
        results = pd.concat([results.drop(["params"], axis=1), results_params], axis=1)
        results["model"] = self.model.name
        results.rename(columns={"target": "rmse"}, inplace=True)
        results["rmse"] = (-1 * results["rmse"]).round(4)
        results.sort_values(by="rmse", ascending=True, inplace=True)
        results.reset_index(drop=True, inplace=True)
        results.to_csv(
            f"{EACH_MODEL_PATH}/{self.model.name}_results_{N_ITER}-iters_{N_SPLITS}-splits.csv",
            index=False,
        )
        
        return results 