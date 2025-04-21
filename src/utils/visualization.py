"""
Visualization utilities.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import learning_curve
import scipy.stats as st
from src.config.configurations import PLOT_DPI, FIGS_PATH, N_ITER, N_SPLITS


class Visualization:
    @staticmethod
    def plot_learning_curve(model, X, y, target_bins, model_name, cv):
        """Plot learning curve."""
        #train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, 
            X, 
            target_bins, 
            cv=cv,  # Use provided cv object
            scoring="r2",
            #train_sizes=train_sizes,
            n_jobs=-1
        )
        
        train_sc_mean = np.mean(train_scores, axis=1)
        train_sc_std = np.std(train_scores, axis=1)
        test_sc_mean = np.mean(test_scores, axis=1)
        test_sc_std = np.std(test_scores, axis=1)
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(train_sizes, train_sc_mean, "o-", color="r", label="Training score", 
            markersize=4, linewidth=1)
        ax.fill_between(
            train_sizes,
            train_sc_mean - train_sc_std,
            train_sc_mean + train_sc_std,
            alpha=0.1,
            color="r"
        )
        ax.plot(train_sizes, test_sc_mean, "o-", color="g", label="Cross-validation score",
            markersize=4, linewidth=1)
        ax.fill_between(
            train_sizes,
            test_sc_mean - test_sc_std,
            test_sc_mean + test_sc_std,
            alpha=0.1,
            color="g"
        )
        ax.set_xlabel("Number of Training Samples", fontsize=10)
        ax.set_ylabel("R² Score", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(False)
        ax.set_title(f"Learning Curve: {model_name}", fontsize=12)
        ax.legend(loc="best", fontsize=8)
        
        plt.tight_layout()
        results_path = os.path.join(FIGS_PATH, f"{model_name}")
        os.makedirs(results_path, exist_ok=True)
        plt.savefig(os.path.join(results_path, f"{model_name}_learning_curve.png"), dpi=PLOT_DPI)
        plt.savefig(os.path.join(results_path, f"{model_name}_learning_curve.pdf"))
        #plt.show()
        plt.close()

    @staticmethod
    def plot_prediction_intervals(df, y_true, y_pred, model_name):
        """Plot prediction intervals."""
        
        mean_residual = np.mean(y_true - y_pred)
        std_residual = np.std(y_true - y_pred, ddof=1)
        confidence_level = 0.95
        z_score = st.norm.ppf(1 - (1 - confidence_level) / 2)
        intervals = z_score * std_residual
        lower_bounds = y_pred - intervals
        lower_bounds = np.clip(lower_bounds, 0, None)
        upper_bounds = y_pred + intervals
        
        plt.figure(figsize=(18, 12))
        plt.plot(df["DEPTH"], y_true, label="True Values", linestyle="--")
        plt.plot(df["DEPTH"], y_pred, label="Predictions", linestyle="-", marker="o")
        plt.fill_between(
            df["DEPTH"],
            lower_bounds,
            upper_bounds,
            color="gray",
            alpha=0.2,
            label="95% Prediction Interval"
        )
        plt.legend(fontsize=16)
        plt.xlabel("Depth (m)", fontsize=20)
        plt.ylabel("TOC (%)", fontsize=20)
        plt.ylim([0, 15])
        plt.title(f"{model_name} Prediction Intervals", fontsize=25)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        results_path = os.path.join(FIGS_PATH, f"{model_name}")
        os.makedirs(results_path, exist_ok=True)
        plt.savefig(os.path.join(results_path, f"{model_name}_prediction_intervals.png"), dpi=PLOT_DPI)
        plt.savefig(os.path.join(results_path, f"{model_name}_prediction_intervals.pdf"))
        #plt.show() 
        plt.close()
        
    @staticmethod
    def plot_prediction_per_well(df, y_true, y_pred, model_name):
        """Plot predictions per well."""
        wells = df["WELLNAME"].unique()
        num_wells = len(wells)
        fig, axes = plt.subplots(1, num_wells, figsize=(num_wells * 2.5, 8), sharey=True)
        
        for i, (ax, well) in enumerate(zip(axes, wells)):
            well_data = df[df["WELLNAME"] == well]
            well_mask = df["WELLNAME"] == well
            well_true = y_true[well_mask]
            well_pred = y_pred[well_mask]
            
            ax.scatter(well_true, well_data["DEPTH"], label="TOC", color="blue")
            ax.plot(well_pred, well_data["DEPTH"], label=f"{model_name} TOC", 
                    color="darkred", linewidth=3)
            ax.set_xlim([0, 15])
            ax.invert_yaxis()
            ax.set_xlabel("TOC (%)", fontsize=15)
            ax.set_title(f"Well {well}", fontsize=15)
            
            if i == num_wells - 1:
                ax.legend(loc="upper right", fontsize="10", markerscale=1)
        
        axes[0].set_ylabel("Depth (m)", fontsize=15)
        
        plt.tight_layout()
        results_path = os.path.join(FIGS_PATH, f"{model_name}")
        os.makedirs(results_path, exist_ok=True)
        plt.savefig(os.path.join(results_path, f"{model_name}_per_well.png"), dpi=PLOT_DPI)
        plt.savefig(os.path.join(results_path, f"{model_name}_per_well.pdf"))
        plt.close()

    @staticmethod
    def plot_ml_comparison_passey(df, y_true, y_pred, model_name, metrics_dict):
        """Plot comparison between ML model and Passey method."""
        import seaborn as sns
        from scipy.stats import pearsonr
        from sklearn import metrics
        
        # Prepare Passey data
        df_passey = df.copy()
        df_passey["TOC_Passey"] = np.where(df_passey["TOC_Passey"] <= 0, np.nan, df_passey["TOC_Passey"])
        df_passey.dropna(subset=["TOC_Passey"], inplace=True)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot Passey comparison
        sns.regplot(x=df_passey["TOC"], y=df_passey["TOC_Passey"], ax=ax1, 
                    scatter_kws={"alpha": 0.5, "s": 4})
        ax1.set_xlabel("Measured TOC (%)", fontsize=15)
        ax1.set_ylabel("Passey TOC (%)", fontsize=15)
        ax1.set_ylim([0, 15])
        
        # Calculate Passey metrics
        r2_passey = metrics.r2_score(df_passey["TOC"], df_passey["TOC_Passey"])
        rmse_passey = metrics.root_mean_squared_error(df_passey["TOC"], df_passey["TOC_Passey"])
        mape_passey = metrics.mean_absolute_percentage_error(df_passey["TOC"], df_passey["TOC_Passey"])
        correlation_passey, _ = pearsonr(df_passey["TOC"], df_passey["TOC_Passey"])
        
        # Add Passey annotations
        ax1.annotate(f"R² = {round(r2_passey, 3)}", xy=(0.95, 0.88), xycoords="axes fraction", 
                    size=13, ha="right")
        ax1.annotate(f"RMSE = {round(rmse_passey, 3)}", xy=(0.95, 0.82), xycoords="axes fraction", 
                    size=13, ha="right")
        ax1.annotate(f"MAPE = {round(mape_passey, 3)}", xy=(0.95, 0.76), xycoords="axes fraction", 
                    size=13, ha="right")
        ax1.annotate("A", xy=(0.00, 1.05), xycoords="axes fraction", size=15, weight="bold")
        ax1.annotate(f"ρ = {round(correlation_passey, 3)}", xy=(0.95, 0.94), 
                    xycoords="axes fraction", size=13, ha="right")
        
        # Plot ML model comparison
        sns.regplot(x=y_true, y=y_pred, ax=ax2, scatter_kws={"alpha": 0.5, "s": 4})
        ax2.set_xlabel("Measured TOC (%)", fontsize=15)
        ax2.set_ylabel(f"{model_name} TOC (%)", fontsize=15)
        ax2.set_ylim([0, 15])
        
        correlation_ml, _ = pearsonr(y_true, y_pred)
        
        # Add ML model annotations
        ax2.annotate(f"R² = {round(metrics_dict['r2'], 3)}", xy=(0.95, 0.88), 
                    xycoords="axes fraction", size=13, ha="right")
        ax2.annotate(f"RMSE = {round(metrics_dict['rmse'], 3)}", xy=(0.95, 0.82), 
                    xycoords="axes fraction", size=13, ha="right")
        ax2.annotate(f"MAPE = {round(metrics_dict['mape'], 3)}", xy=(0.95, 0.76), 
                    xycoords="axes fraction", size=13, ha="right")
        ax2.annotate("B", xy=(0.00, 1.05), xycoords="axes fraction", size=15, weight="bold")
        ax2.annotate(f"ρ = {round(correlation_ml, 3)}", xy=(0.95, 0.94), 
                    xycoords="axes fraction", size=13, ha="right")
        
        plt.tight_layout()
        results_path = os.path.join(FIGS_PATH, f"{model_name}")
        os.makedirs(results_path, exist_ok=True)
        plt.savefig(os.path.join(results_path, f"{model_name}_passey_comparison.png"), dpi=PLOT_DPI)
        plt.savefig(os.path.join(results_path, f"{model_name}_passey_comparison.pdf"))
        plt.close()
        
    @staticmethod
    def plot_models_comparison(model_names):
        """Plot comparison between all trained models."""
        
        results = {}
        base_path = os.path.join(FIGS_PATH)
        
        # Load data for each model
        for model_name in model_names:
            model_path = os.path.join(FIGS_PATH, model_name, f"{model_name}_final_df_{N_SPLITS}-skf.csv")
            if os.path.exists(model_path):
                df = pd.read_csv(model_path)
                results[model_name] = df
            else:
                print(f"Warning: No results found for {model_name} at {model_path}")
        
        if not results:
            print("No model results found to compare")
            return
        
        # Create combined dataframe
        df_base = results[model_names[0]]
        df_results = df_base[["WELLNAME", "DEPTH", "TOC"]].copy()
        
        # Add predictions from each model
        for model_name in model_names:
            df_results[f"TOC_{model_name}"] = results[model_name][f"TOC_{model_name}"]
        
        # Add measurement error
        error_abs = 0.3  # Estimated error in TOC measurements
        df_results['TOC_error'] = error_abs
        
        # Create plot
        fig, axes = plt.subplots(1, len(model_names), figsize=(6*len(model_names), 18), sharey=True)
        
        # Plot each model
        for i, (ax, model_name) in enumerate(zip(axes, model_names)):
            # Plot error bars and measurements
            ax.errorbar(df_results["TOC"], df_results["DEPTH"], 
                    xerr=df_results['TOC_error'],
                    fmt='none', ecolor='gray', capsize=3, 
                    label='_nolegend_')
            ax.scatter(df_results["TOC"], df_results["DEPTH"], 
                    label="TOC (%) with Error", alpha=1)
            
            # Plot predictions
            ax.plot(df_results[f"TOC_{model_name}"], df_results["DEPTH"], 
                    color="darkred", label=f"{model_name} (%)", linewidth=3)
            
            # Customize plot
            ax.set_xlim([0, 15])
            ax.set_xlabel("TOC (%)", fontsize=25)
            if i == 0:
                ax.set_ylabel("Depth (m)", fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.legend(loc="upper right", fontsize=25, markerscale=1)
            ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save plot
        comparison_path = os.path.join(FIGS_PATH, "comparison")
        os.makedirs(comparison_path, exist_ok=True)
        plt.savefig(os.path.join(comparison_path, "TOC_models_comparison.png"), dpi=PLOT_DPI)
        plt.savefig(os.path.join(comparison_path, "TOC_models_comparison.pdf"))
        plt.close()
        
        # Calculate and print RMSE scores per well
        wells = df_results["WELLNAME"].unique()
        well_metrics = {}
        
        for well in wells:
            well_data = df_results[df_results["WELLNAME"] == well]
            well_metrics[well] = {}
            
            for model_name in model_names:
                rmse = metrics.root_mean_squared_error(
                    well_data["TOC"], 
                    well_data[f"TOC_{model_name}"]
                )
                well_metrics[well][model_name] = rmse
        
        # Save metrics to file
        metrics_df = pd.DataFrame(well_metrics).T
        metrics_df.to_csv(os.path.join(comparison_path, "well_rmse_metrics.csv"))
        
        return well_metrics