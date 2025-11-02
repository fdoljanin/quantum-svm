import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from qsvm.config.types import DataConfig

class DataPipeline:
    """
    Data loading and preprocessing pipeline for quantum SVM experiments.

    Handles:
    - Loading compressed CSV data
    - Feature/target selection
    - Missing value handling
    - MinMaxScaler normalization
    - Additional scaling transformations
    - Train/test splitting
    """

    def __init__(self, config: DataConfig):
        """
        Initialize data pipeline with configuration.

        Args:
            config: DataConfig specifying data source and preprocessing
        """
        self.config = config
        self.scaler = MinMaxScaler(feature_range=config.scale_range)
        self.data_raw = None
        self.data_features = None
        self.data_target = None

    @classmethod
    def from_config(cls, config: DataConfig) -> "DataPipeline":
        """Create pipeline from configuration."""
        return cls(config)

    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and clean raw data.

        Returns:
            Tuple of (features_df, target_array)
        """
        # Load CSV
        self.data_raw = pd.read_csv(
            self.config.data_path,
            nrows=self.config.nrows,
            header=None,
        )

        # Drop missing values
        cols_to_check = [self.config.target_column] + list(self.config.feature_columns)
        self.data_raw = self.data_raw.dropna(subset=cols_to_check)

        return self.data_raw

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data: scale features and extract target.

        Returns:
            Tuple of (features_array, target_array)
        """
        if self.data_raw is None:
            self.load_data()

        # Extract and scale features
        features_raw = self.data_raw[list(self.config.feature_columns)]
        self.data_features = self.scaler.fit_transform(features_raw)

        # Apply additional scaling factor
        self.data_features = self.data_features * self.config.scale_factor

        # Extract target
        self.data_target = self.data_raw[self.config.target_column].to_numpy()

        return self.data_features, self.data_target

    def split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.

        Returns:
            Tuple of (x_train, y_train, x_test, y_test)
        """
        if self.data_features is None or self.data_target is None:
            self.preprocess()

        # Train: first N samples
        x_train = self.data_features[:self.config.train_size]
        y_train = self.data_target[:self.config.train_size]

        # Test: last N samples
        x_test = self.data_features[-self.config.test_size:]
        y_test = self.data_target[-self.config.test_size:]

        return x_train, y_train, x_test, y_test

    def load_and_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience method: load, preprocess, and split in one call.

        Returns:
            Tuple of (x_train, y_train, x_test, y_test)
        """
        self.load_data()
        self.preprocess()
        return self.split()

    def get_feature_range(self) -> Tuple[float, float]:
        """Get actual feature range after preprocessing."""
        if self.data_features is None:
            self.preprocess()
        return float(self.data_features.min()), float(self.data_features.max())
