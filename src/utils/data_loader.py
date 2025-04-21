"""
Data loading and preprocessing utilities.
"""
import pandas as pd
import math
from src.config.configurations import N_SPLITS

class DataLoader:
    @staticmethod
    def load_data(data_path):
        """Load and preprocess data."""
        df = pd.read_pickle(data_path)
        df.rename(columns={"COT": "TOC"}, inplace=True)
        
        # Create target bins for stratified k-fold
        num_bins = math.floor(len(df) / N_SPLITS)
        bins_on = df["TOC"]
        qc = pd.cut(bins_on.tolist(), num_bins)
        df.loc[:, "target_bins"] = qc.codes
        
        return df
    
    @staticmethod
    def filter_wells(df, wells):
        """Filter data for specific wells."""
        if wells is not None:
            df = df[df["WELLNAME"].isin(wells)].copy()
            df.reset_index(drop=True, inplace=True)
        return df 