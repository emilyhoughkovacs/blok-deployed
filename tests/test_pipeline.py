"""Unit and smoke tests for the persona clustering pipeline."""

import numpy as np
import pandas as pd
import pytest

from persona_clustering.config import (
    RAW_FEATURES,
    CLUSTERING_FEATURES,
    LOG_TRANSFORM_FEATURES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_raw_features():
    """Small DataFrame with 100 synthetic customers, 9 raw features."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "frequency": np.random.choice([1, 1, 1, 2, 3], n),
            "monetary_total": np.random.lognormal(4.5, 0.9, n),
            "monetary_avg_item": np.random.lognormal(4.3, 0.9, n),
            "avg_items_per_order": np.random.choice([1.0, 1.0, 1.5, 2.0], n),
            "avg_installments": np.random.uniform(0, 10, n),
            "pct_credit_card": np.random.choice([0.0, 0.5, 1.0], n),
            "category_diversity": np.random.choice([1, 1, 1, 2, 3], n),
            "is_positive_reviewer": np.random.choice([0, 1], n, p=[0.2, 0.8]),
            "is_weekend_shopper": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        },
        index=pd.Index(
            [f"cust_{i}" for i in range(n)], name="customer_unique_id"
        ),
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_feature_lists_consistent(self):
        assert len(RAW_FEATURES) == len(CLUSTERING_FEATURES) == 9

    def test_log_transform_features_subset_of_raw(self):
        assert set(LOG_TRANSFORM_FEATURES).issubset(set(RAW_FEATURES))

    def test_project_root_exists(self):
        from persona_clustering.config import PROJECT_ROOT

        assert PROJECT_ROOT.exists()


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    def test_apply_transforms_shape(self, sample_raw_features):
        from persona_clustering.features.engineering import apply_transforms

        result = apply_transforms(sample_raw_features)
        assert result.shape == (100, 13)

    def test_apply_transforms_log_columns_present(self, sample_raw_features):
        from persona_clustering.features.engineering import apply_transforms

        result = apply_transforms(sample_raw_features)
        for f in LOG_TRANSFORM_FEATURES:
            assert f"{f}_log" in result.columns

    def test_apply_transforms_no_nans(self, sample_raw_features):
        from persona_clustering.features.engineering import apply_transforms

        result = apply_transforms(sample_raw_features)
        assert result.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Clustering tests
# ---------------------------------------------------------------------------


class TestClustering:
    def test_scale_features_zero_mean(self, sample_raw_features):
        from persona_clustering.features.engineering import apply_transforms
        from persona_clustering.models.clustering import scale_features

        transformed = apply_transforms(sample_raw_features)[CLUSTERING_FEATURES]
        X_scaled, scaler = scale_features(transformed)
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-6)

    def test_fit_kmeans_returns_labels(self, sample_raw_features):
        from persona_clustering.features.engineering import apply_transforms
        from persona_clustering.models.clustering import scale_features, fit_kmeans

        transformed = apply_transforms(sample_raw_features)[CLUSTERING_FEATURES]
        X_scaled, _ = scale_features(transformed)
        model, labels = fit_kmeans(X_scaled, n_clusters=3)
        assert len(labels) == 100
        assert set(labels) == {0, 1, 2}

    def test_run_returns_clustering_result(self, sample_raw_features):
        from persona_clustering.features.engineering import apply_transforms
        from persona_clustering.models.clustering import run

        transformed = apply_transforms(sample_raw_features)[CLUSTERING_FEATURES]
        result = run(transformed, n_clusters=3)
        assert len(result.labels) == 100
        assert result.silhouette > -1


# ---------------------------------------------------------------------------
# Persona profiling tests
# ---------------------------------------------------------------------------


class TestProfiler:
    def test_compute_cluster_centroids_shape(self, sample_raw_features):
        from persona_clustering.features.engineering import apply_transforms
        from persona_clustering.models.clustering import run as cluster_run
        from persona_clustering.personas.profiler import compute_cluster_centroids

        transformed = apply_transforms(sample_raw_features)[CLUSTERING_FEATURES]
        cluster_result = cluster_run(transformed, n_clusters=3)
        centroids = compute_cluster_centroids(
            sample_raw_features, cluster_result.labels
        )
        assert centroids.shape == (3, 9)

    def test_describe_feature_returns_string(self):
        from persona_clustering.personas.profiler import describe_feature

        pop_stats = pd.DataFrame(
            {"frequency": [1.03, 1.0, 0.21]}, index=["mean", "median", "std"]
        )
        result = describe_feature("frequency", 2.1, 5.1, pop_stats)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_run_produces_correct_persona_count(self, sample_raw_features):
        from persona_clustering.features.engineering import apply_transforms
        from persona_clustering.models.clustering import run as cluster_run
        from persona_clustering.personas.profiler import run as profiler_run

        transformed = apply_transforms(sample_raw_features)[CLUSTERING_FEATURES]
        cluster_result = cluster_run(transformed, n_clusters=3)
        persona_set = profiler_run(sample_raw_features, cluster_result.labels)
        assert len(persona_set.personas) == 3
        for pid, persona in persona_set.personas.items():
            assert "persona_name" in persona
            assert "agent_system_prompt" in persona


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------


class TestAgent:
    def test_persona_agent_mock_response(self):
        from persona_clustering.personas.agent import PersonaAgent

        agent = PersonaAgent(
            cluster_id=0,
            persona_name="Mainstream Shopper",
            system_prompt="You are a test persona.",
            mock_mode=True,
        )
        response = agent.respond("test scenario")
        assert isinstance(response, str)
        assert "MOCK" in response

    def test_validate_persona_consistency(self):
        from persona_clustering.personas.agent import validate_persona_consistency

        mock_results = pd.DataFrame(
            {
                "persona_name": ["Cash Customer"],
                "scenario_name": ["financing_offer"],
                "decision": ["No"],
                "response": [
                    "I prefer to pay upfront with boleto and avoid credit card debt."
                ],
            }
        )
        validations = validate_persona_consistency(mock_results)
        assert len(validations) >= 1
        assert validations[0]["aligned"] is True


# ---------------------------------------------------------------------------
# Smoke test (requires raw data on disk)
# ---------------------------------------------------------------------------


class TestPipelineSmoke:
    @pytest.fixture(autouse=True)
    def check_data_exists(self):
        from persona_clustering.config import RAW_DATA_DIR

        if not (RAW_DATA_DIR / "olist_customers_dataset.csv").exists():
            pytest.skip("Raw data not available")

    def test_loader_runs(self):
        from persona_clustering.data.loader import run

        raw_data = run(save=False)
        assert raw_data.master_df.shape[0] > 100000

    def test_full_pipeline_no_save(self):
        from persona_clustering.data.loader import run as load_run
        from persona_clustering.features.engineering import run as feat_run
        from persona_clustering.models.clustering import run as clust_run
        from persona_clustering.personas.profiler import run as prof_run

        raw = load_run(save=False)
        features = feat_run(raw, save=False)
        clusters = clust_run(features.transformed_features, save=False)
        personas = prof_run(features.raw_features, clusters.labels, save=False)
        assert len(personas.personas) == 7
