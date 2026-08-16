"""One-hot design matrix builder shared by the GLM frequency and severity fits.

Categories are fixed from the training data at fit time and reapplied at
predict time, so a level that doesn't happen to appear in a given batch of
new-business quotes doesn't shift the column layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from moto_pricing.features import CATEGORICAL_FEATURES


@dataclass
class DesignMatrixBuilder:
    features: list[str]
    categories: dict[str, list[str]] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "DesignMatrixBuilder":
        cat_features = [f for f in self.features if f in CATEGORICAL_FEATURES]
        self.categories = {f: sorted(df[f].astype(str).unique()) for f in cat_features}
        self.columns = list(self.transform(df).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = [pd.Series(1.0, index=df.index, name="const")]
        for feat in self.features:
            if feat in self.categories:
                cat = pd.Categorical(
                    df[feat].astype(str), categories=self.categories[feat]
                )
                dummies = pd.get_dummies(cat, prefix=feat, drop_first=True, dtype=float)
                dummies.index = df.index
                parts.append(dummies)
            else:
                parts.append(df[feat].astype(float).rename(feat))
        X = pd.concat(parts, axis=1)
        if self.columns:
            X = X.reindex(columns=self.columns, fill_value=0.0)
        return X


def log_offset(exposure: pd.Series) -> np.ndarray:
    """log(exposure), clipped away from zero for a safe offset."""
    return np.log(np.clip(exposure.to_numpy(dtype=float), 1e-6, None))
