"""
Multi-layer Perceptron model implementation.
"""
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel

class MLPModel(BaseModel):
    def __init__(self):
        super().__init__("MLP")
        self.nn_activation_map = {0: "tanh", 1: "relu"}
        self.nn_learning_rate_map = {0: "constant", 1: "invscaling", 2: "adaptive"}
        self.nn_hidden_layer_sizes_map = {0: (25,), 1: (100,), 2: (100, 5)}
        
    def create_model(self, **params):
        """Create MLP model with given parameters."""
        # Map the parameters
        hidden_layer_sizes = self.nn_hidden_layer_sizes_map[int(round(params.get("hidden_layer_sizes", 0)))]
        activation = self.nn_activation_map[int(round(params.get("activation", 0)))]
        learning_rate = self.nn_learning_rate_map[int(round(params.get("learning_rate", 0)))]
        max_iter = int(round(params.get("max_iter", 1000)))
        alpha = params.get("alpha", 0.0001)
        tol = params.get("tol", 1e-4)
        
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    alpha=alpha,
                    tol=tol,
                    random_state=42,
                ),
            ),
        ])
        return self.model 