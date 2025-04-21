"""
Base model class that defines the interface for all models.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore', message='The least populated class in y has only')

from src.config.configurations import N_SPLITS, N_ITER, FIGS_PATH, PLOT_FIGSIZE, PLOT_DPI

class BaseModel(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None
        self.results_path = os.path.join(FIGS_PATH, f"{name}")
        self._create_results_dir()

    def _create_results_dir(self):
        """Create results dir if it doesn't exist."""
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

    @abstractmethod
    def create_model(self, **params):
        """Create the model with given parameters."""
        pass

    def prepare_data(self, df, feature_columns):
        """Prepare data for training."""
        X = df[feature_columns].copy()
        y = df["TOC"]
        target_bins = df["target_bins"]
        return X, y, target_bins

    def evaluate_model(self, model, X, y, df):
        """Evaluate model performance."""
        n_splits = N_SPLITS
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_num = 0
        r2_scores, rmse_scores, mae_scores, mape_scores, residuals_25 = [], [], [], [], []
        errors_df = pd.DataFrame()
        
        all_y_test = np.zeros(len(df))
        all_predictions = np.zeros(len(df))
        
        fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=PLOT_FIGSIZE)
        plt.subplots_adjust(left=0.15)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_splits))
        
        for fold_idx, (train_index, test_index) in enumerate(self.skf.split(X, df["target_bins"])):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            self.model.fit(X_train, y_train)
            test_data_prediction = self.model.predict(X_test)
            
            all_predictions[test_index] = test_data_prediction
            all_y_test[test_index] = y_test
            
            r2_score = metrics.r2_score(y_test, test_data_prediction)
            rmse_score = metrics.root_mean_squared_error(y_test, test_data_prediction)
            mae_score = metrics.mean_absolute_error(y_test, test_data_prediction)
            mape_score = metrics.mean_absolute_percentage_error(y_test, test_data_prediction)
            
            r2_scores.append(np.round(r2_score, 3))
            rmse_scores.append(np.round(rmse_score, 3))
            mae_scores.append(np.round(mae_score, 3))
            mape_scores.append(np.round(mape_score, 3))
            
            fold_num += 1
            print(f"Fold number {fold_num}: RMSE = {np.round(rmse_score, 3)}")
            
            # Plot with colors
            ax1.scatter(y_test, test_data_prediction, label=f"Fold {fold_num}", alpha=0.5, s=4, color=colors[fold_idx])
            ax1.set_xlim(0, 15)
            ax1.set_ylim(0, 15)
            
            percentual_residual_error = (y_test - test_data_prediction) / y_test
            ranges = [-0.25, 0, 0.25]
            low_residuals = (
                percentual_residual_error.groupby(pd.cut(percentual_residual_error, ranges), observed=True).count().sum()
            )
            low_residuals_percentage = 100 * low_residuals / percentual_residual_error.count()
            residuals_25.append(low_residuals_percentage)
            
            residual_error = y_test - test_data_prediction
            errors_df[f"y_test_{fold_num}"] = y_test
            errors_df[f"y_residuals_{fold_num}"] = percentual_residual_error
            ax2.scatter(y_test, residual_error, label=f"Fold {fold_num}", alpha=0.5, s=4, color=colors[fold_idx])
            ax2.axhline(0, color="r", linestyle="--", alpha=0.5, linewidth=0.8)
            ax2.set_xlim(0, 15)
            
            print(f"Fold number {fold_num}: Data inside the ± 25% error range = {np.round(low_residuals_percentage, 3)}%")
            
            ax3.scatter(y_test, percentual_residual_error, label=f"Fold {fold_num}", alpha=0.5, s=4, color=colors[fold_idx])
            ax3.set_xlim(0, 15)
        
        # Add RMSE stats to first panel
        avg_rmse = np.mean(rmse_scores)
        min_rmse = np.min(rmse_scores)
        max_rmse = np.max(rmse_scores)
        ax1.annotate(f"max. RMSE = {max_rmse:.3f}", xy=(0.95, 0.94), xycoords="axes fraction", size=15, ha="right")
        ax1.annotate(f"min. RMSE = {min_rmse:.3f}", xy=(0.95, 0.88), xycoords="axes fraction", size=15, ha="right")
        ax1.annotate(f"avg. RMSE = {avg_rmse:.3f}", xy=(0.95, 0.82), xycoords="axes fraction", size=15, ha="right")
        
        # Add residuals info to last panel
        avg_residuals_25 = np.mean(residuals_25)
        ax3.annotate(f"Data inside ±25% = {avg_residuals_25:.2f}%", xy=(0.2, 0.75), xycoords="axes fraction", size=15)
        ax3.fill_between(x=plt.gca().get_xlim(), y1=0.25, y2=-0.25, color="black", alpha=0.25)
        
        # Add legend of the folds outside the plot
        ax3.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="16", markerscale=1)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f"{self.name}_combined_{N_SPLITS}-skf.png"), dpi=PLOT_DPI)
        plt.savefig(os.path.join(self.results_path, f"{self.name}_combined_{N_SPLITS}-skf.pdf"))
        #plt.show()
        plt.close()
        
        
        df_results = df.copy()
        df_results[f"TOC_{self.name}"] = all_predictions
        df_results[f"Error_{self.name}"] = df_results["TOC"] - df_results[f"TOC_{self.name}"]
        df_results.sort_values(by="DEPTH", inplace=True)
        
        # Save the DataFrame
        df_results.to_csv(
            os.path.join(self.results_path, f"{self.name}_final_df_{N_SPLITS}-skf.csv"),
            index=False
        )
        
        return {
            "all_y_test": all_y_test,
            "all_predictions": all_predictions,
            "r2_scores": r2_scores,
            "rmse_scores": rmse_scores,
            "mae_scores": mae_scores,
            "mape_scores": mape_scores,
            "residuals_25": residuals_25,
            "errors_df": errors_df,
            "fig": fig,
            "ax1": ax1,
            "ax2": ax2,
            "ax3": ax3
        } 