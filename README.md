# TOC Prediction Project

This project contains scripts for the paper "TOC Prediction from Well Logs Using Gradient Boosting and Neural Network in the Santos Basin, SE Brazil" submitted to the Marine and Petroleum Geology Journal. 

SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5211368

## Environment Setup

You can set up the environment using either Conda or Python's virtual environment (venv).

### Option 1: Using Conda

1. Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://www.anaconda.com/download) if you haven't already.

2. Create a new conda environment with Python 3.11:
```bash
conda create -n toc_prediction python=3.11
conda activate toc_prediction
```

2. Using the environment.yml file:
```bash
conda env create -f environment.yml
```

3. Install required packages (if you didn't use the environment.yml file):
```bash
pip install -r requirements.txt
```

### Option 2: Using Python venv

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
- On Windows:
```bash
.venv\Scripts\activate
```
- On Unix or MacOS:
```bash
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/
│   └── df_5_wells_fe.pkl
├── results/
│   └── experiment_name/
│       ├── each_model/
│       ├── figs/
│       ├── hpt_tuning/
│       └── rmse_results/
└── src/
    ├── config/
    │   └── configurations.py    # All configurable parameters
    ├── models/
    │   ├── base_model.py       # Base class for all models
    │   ├── gbdt.py            # GBDT model implementation
    │   ├── mlp.py             # MLP model implementation
    │   └── xgb.py             # XGBoost model implementation
    ├── training/
    │   ├── train.py           # Main training pipeline
    │   └── hyperparameter_tuning.py
    ├── utils/
    │   ├── data_loader.py
    │   ├── metrics.py
    │   └── visualization.py
    ├── main.py                # Entry point    
    └── requirements.txt
```

## Configuration

All configurations are centralized in `src/config/configurations.py`. The most important parameters are:

```python
# Model training settings
EXPERIMENT_NAME = "experiment_1"  # Name of the experiment (affects output directory)
N_ITER = 500                     # Number of iterations for Bayesian optimization
N_SPLITS = 10                    # Number of splits for cross-validation
```

Other configurable parameters include:
- Data and results paths
- Feature columns and well selection
- Plot settings
- Bayesian optimization parameters
- Model-specific hyperparameter bounds

While the default values should work well, you can modify any of these parameters to experiment with different settings.

## Usage

Simply run the main script:
```bash
python main.py
```

This will:
1. Train all configured models with hyperparameter optimization
2. Generate performance plots and metrics
3. Save results in the specified experiment directory

## Adding New Models

To add a new model:

1. Create a new model class in `src/models/` that inherits from `BaseModel`:
```python
from src.models.base_model import BaseModel

class NewModel(BaseModel):
    def __init__(self):
        super().__init__("NEW_MODEL_NAME")
    
    def create_model(self, **params):
        # Implement model creation
        pass
```

2. Add hyperparameter bounds in `configurations.py`:
```python
NEW_MODEL_PBOUNDS = {
    "param1": (min_value, max_value),
    "param2": (min_value, max_value),
}
```

3. Add the model to the trainers list in `main.py`:
```python
trainers = [
    (XGBModel, XGB_PBOUNDS),
    (GBDTModel, GBDT_PBOUNDS),
    (MLPModel, MLP_PBOUNDS),
    (NewModel, NEW_MODEL_PBOUNDS),  # Add your new model here
]
```

## Requirements

The project dependencies are managed through either:
- `environment.yml` (for Conda users)
- `requirements.txt` (for pip users)

Key dependencies include:
- Python 3.11
- pandas==2.2.3
- numpy==2.1.3
- scikit-learn==1.6.1
- scipy==1.15.2
- notebook==7.3.2
- matplotlib==3.10.1
- seaborn==0.13.2
- xgboost==2.1.4
- bayesian-optimization==1.5.1
