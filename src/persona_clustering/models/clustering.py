"""StandardScaler + KMeans clustering pipeline.

Extracted from NB3_clustering.ipynb cells 7-24.
"""

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from persona_clustering.config import (
    OPTIMAL_K,
    RANDOM_STATE,
    N_INIT,
    K_RANGE,
    PROCESSED_DATA_DIR,
)


@dataclass
class ClusteringResult:
    """Container for clustering stage outputs."""

    labels: pd.Series
    scaler: StandardScaler
    model: KMeans
    X_scaled: np.ndarray
    silhouette: float
    inertia: float


def scale_features(
    features_df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Fit (or apply) StandardScaler.  (NB3 cell 7)"""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df)
    else:
        X_scaled = scaler.transform(features_df)
    return X_scaled, scaler


def evaluate_k_range(
    X_scaled: np.ndarray,
    k_range: range = K_RANGE,
    random_state: int = RANDOM_STATE,
    n_init: int = N_INIT,
) -> pd.DataFrame:
    """Run KMeans for each k, return inertia + silhouette scores.  (NB3 cell 9)"""
    results = {"k": [], "inertia": [], "silhouette": []}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = kmeans.fit_predict(X_scaled)
        results["k"].append(k)
        results["inertia"].append(kmeans.inertia_)
        results["silhouette"].append(silhouette_score(X_scaled, labels))

    return pd.DataFrame(results)


def fit_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = OPTIMAL_K,
    random_state: int = RANDOM_STATE,
    n_init: int = N_INIT,
) -> tuple[KMeans, np.ndarray]:
    """Fit final KMeans model and return (model, labels).  (NB3 cell 15)"""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X_scaled)
    return model, labels


def run(
    transformed_features: pd.DataFrame,
    n_clusters: int = OPTIMAL_K,
    evaluate_range: bool = False,
    save: bool = True,
) -> ClusteringResult:
    """Pipeline entry point: scale → (optionally evaluate k) → fit → return."""
    X_scaled, scaler = scale_features(transformed_features)

    if evaluate_range:
        eval_df = evaluate_k_range(X_scaled)
        print(eval_df.to_string(index=False))

    model, labels = fit_kmeans(X_scaled, n_clusters=n_clusters)
    sil = silhouette_score(X_scaled, labels)

    labels_series = pd.Series(
        labels, index=transformed_features.index, name="cluster"
    )

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        labels_series.to_csv(PROCESSED_DATA_DIR / "customer_clusters.csv")
        joblib.dump(scaler, PROCESSED_DATA_DIR / "scaler.pkl")
        joblib.dump(model, PROCESSED_DATA_DIR / "kmeans_model.pkl")

    return ClusteringResult(
        labels=labels_series,
        scaler=scaler,
        model=model,
        X_scaled=X_scaled,
        silhouette=sil,
        inertia=model.inertia_,
    )
