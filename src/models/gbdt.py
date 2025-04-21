"""
Gradient Boosting Decision Tree model implementation.
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel

class GBDTModel(BaseModel):
    def __init__(self):
        super().__init__("GBDT")
        
    def create_model(self, **params):
        """Create GBDT model with given parameters."""
        # Convert integer parameters
        n_estimators = int(round(params.get("n_estimators", 100)))
        max_depth = int(round(params.get("max_depth", 3)))
        min_samples_split = int(round(params.get("min_samples_split", 2)))
        min_samples_leaf = int(round(params.get("min_samples_leaf", 1)))
        max_features = params.get("max_features", None)
        learning_rate = params.get("learning_rate", 0.1)
        tol = params.get("tol", 1e-4)
        
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    learning_rate=learning_rate,
                    tol=tol,
                    random_state=42,
                ),
            ),
        ])
        return self.model 