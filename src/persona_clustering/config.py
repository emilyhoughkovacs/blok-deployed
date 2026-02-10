"""Centralized configuration — paths, feature lists, hyperparameters.

Values extracted from notebooks NB2–NB5 and src/agents.py so that every
module imports from one place instead of hardcoding constants.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current


PROJECT_ROOT = _find_project_root()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Raw dataset filenames  (NB2 cells 4-6)
# ---------------------------------------------------------------------------

RAW_DATASETS: dict[str, str] = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
}

# Date columns that need datetime parsing  (NB1/NB2)
ORDERS_DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
REVIEWS_DATE_COLS = [
    "review_creation_date",
    "review_answer_timestamp",
]

# ---------------------------------------------------------------------------
# Feature engineering  (NB2 cells 11-34)
# ---------------------------------------------------------------------------

RAW_FEATURES = [
    "frequency",
    "monetary_total",
    "monetary_avg_item",
    "avg_items_per_order",
    "avg_installments",
    "pct_credit_card",
    "category_diversity",
    "is_positive_reviewer",
    "is_weekend_shopper",
]

LOG_TRANSFORM_FEATURES = [
    "frequency",
    "monetary_total",
    "monetary_avg_item",
    "avg_items_per_order",
]

CLUSTERING_FEATURES = [
    f"{f}_log" if f in LOG_TRANSFORM_FEATURES else f for f in RAW_FEATURES
]

# ---------------------------------------------------------------------------
# Clustering hyperparameters  (NB3 cells 9-14)
# ---------------------------------------------------------------------------

K_RANGE = range(2, 11)
OPTIMAL_K = 7
RANDOM_STATE = 42
N_INIT = 10

# ---------------------------------------------------------------------------
# Persona profiling  (NB4 cell 15)
# ---------------------------------------------------------------------------

Z_SCORE_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Agent / LLM settings  (src/agents.py)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-20250514"
