"""
Metrics calculation utilities.
"""
import numpy as np
from sklearn import metrics
import pandas as pd

class Metrics:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate various regression metrics."""
        r2 = metrics.r2_score(y_true, y_pred)
        rmse = metrics.root_mean_squared_error(y_true, y_pred)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
        
        return {
            "r2": np.round(r2, 3),
            "rmse": np.round(rmse, 3),
            "mae": np.round(mae, 3),
            "mape": np.round(mape, 3)
        }
    
    @staticmethod
    def calculate_residuals(y_true, y_pred):
        """Calculate residual errors."""
        absolute_error = y_true - y_pred
        percentual_error = absolute_error / y_true
        
        return {
            "absolute": absolute_error,
            "percentual": percentual_error
        }
    
    @staticmethod
    def calculate_error_range(percentual_error, range_value=0.25):
        """Calculate percentage of predictions within error range."""
        ranges = [-range_value, 0, range_value]
        low_residuals = (
            percentual_error.groupby(pd.cut(percentual_error, ranges), observed=True).count().sum()
        )
        return 100 * low_residuals / percentual_error.count() 