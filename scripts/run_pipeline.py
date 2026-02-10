#!/usr/bin/env python
"""End-to-end pipeline: data -> features -> clusters -> personas."""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from persona_clustering.data import loader
from persona_clustering.features import engineering
from persona_clustering.models import clustering
from persona_clustering.personas import profiler
from persona_clustering.config import OPTIMAL_K


def main(
    data_dir: Path | None = None,
    n_clusters: int = OPTIMAL_K,
    evaluate_k: bool = False,
    save: bool = True,
):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger(__name__)

    log.info("Stage 1/4: Loading raw data...")
    raw_data = loader.run(data_dir=data_dir)
    log.info(f"  Master table: {raw_data.master_df.shape}")

    log.info("Stage 2/4: Engineering features...")
    feature_data = engineering.run(raw_data, save=save)
    log.info(f"  Feature matrix: {feature_data.raw_features.shape}")

    log.info("Stage 3/4: Clustering...")
    cluster_result = clustering.run(
        feature_data.transformed_features,
        n_clusters=n_clusters,
        evaluate_range=evaluate_k,
        save=save,
    )
    log.info(f"  Silhouette score: {cluster_result.silhouette:.4f}")
    log.info(
        f"  Cluster sizes: "
        f"{cluster_result.labels.value_counts().sort_index().to_dict()}"
    )

    log.info("Stage 4/4: Generating personas...")
    persona_set = profiler.run(
        feature_data.raw_features,
        cluster_result.labels,
        save=save,
    )
    log.info(f"  Generated {len(persona_set.personas)} personas")

    for cid, p in sorted(persona_set.personas.items()):
        log.info(f"    Cluster {cid}: {p['persona_name']} ({p['size']:,} customers)")

    log.info("Pipeline complete.")
    return persona_set


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run persona clustering pipeline")
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Override raw data directory"
    )
    parser.add_argument(
        "--n-clusters", type=int, default=OPTIMAL_K, help="Number of clusters"
    )
    parser.add_argument(
        "--evaluate-k", action="store_true", help="Run k selection analysis"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving intermediate files"
    )
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        n_clusters=args.n_clusters,
        evaluate_k=args.evaluate_k,
        save=not args.no_save,
    )
