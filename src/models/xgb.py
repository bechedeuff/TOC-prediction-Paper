"""
XGBoost model implementation.
"""
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel

class XGBModel(BaseModel):
    def __init__(self):
        super().__init__("XGB")
        
    def create_model(self, **params):
        """Create XGBoost model with given parameters."""
        # Convert integer parameters
        n_estimators = int(round(params.get("n_estimators", 100)))
        max_depth = int(round(params.get("max_depth", 3)))
        min_child_weight = int(round(params.get("min_child_weight", 1)))
        eta = params.get("eta", 0.3)
        subsample = params.get("subsample", 1.0)
        colsample_bytree = params.get("colsample_bytree", 1.0)
        reg_alpha = params.get("reg_alpha", 0)
        reg_lambda = params.get("reg_lambda", 1)
        
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    eta=eta,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    random_state=42,
                ),
            ),
        ])
        return self.model 