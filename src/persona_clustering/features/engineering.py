"""Compute the 9 behavioral features and apply log transforms.

Extracted from NB2_feature_engineering.ipynb cells 11-34.
Each compute_*() function corresponds to one feature; build_feature_matrix()
assembles them all into a single DataFrame indexed by customer_unique_id.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from persona_clustering.config import (
    RAW_FEATURES,
    LOG_TRANSFORM_FEATURES,
    CLUSTERING_FEATURES,
    PROCESSED_DATA_DIR,
)
from persona_clustering.data.loader import RawData


@dataclass
class FeatureData:
    """Container for the feature engineering stage outputs."""

    raw_features: pd.DataFrame        # 93K × 9 (original scale)
    transformed_features: pd.DataFrame  # 93K × 9 (clustering-ready)
    complete_features: pd.DataFrame     # 93K × 13 (raw + log cols)


# ---------------------------------------------------------------------------
# Individual feature computations  (NB2 cells 11-27)
# ---------------------------------------------------------------------------


def compute_frequency(
    delivered_orders: pd.DataFrame, customers: pd.DataFrame
) -> pd.Series:
    """Order count per customer.  (NB2 cell 11)"""
    return (
        delivered_orders.merge(customers, on="customer_id")
        .groupby("customer_unique_id")["order_id"]
        .nunique()
        .rename("frequency")
    )


def compute_monetary_total(master_df: pd.DataFrame) -> pd.Series:
    """Lifetime spend per customer.  (NB2 cell 13)"""
    return (
        master_df.groupby("customer_unique_id")["price"]
        .sum()
        .rename("monetary_total")
    )


def compute_monetary_avg_item(master_df: pd.DataFrame) -> pd.Series:
    """Average item price per customer.  (NB2 cell 15)"""
    item_agg = master_df.groupby("customer_unique_id").agg(
        total_spend=("price", "sum"),
        total_items=("order_item_id", "count"),
    )
    return (item_agg["total_spend"] / item_agg["total_items"]).rename(
        "monetary_avg_item"
    )


def compute_avg_items_per_order(master_df: pd.DataFrame) -> pd.Series:
    """Mean basket size across orders.  (NB2 cell 17)"""
    items_per_order = (
        master_df.groupby(["customer_unique_id", "order_id"])
        .size()
        .rename("items_in_order")
    )
    return (
        items_per_order.groupby("customer_unique_id")
        .mean()
        .rename("avg_items_per_order")
    )


def compute_avg_installments(
    delivered_orders: pd.DataFrame,
    customers: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.Series:
    """Mean payment installments per customer.  (NB2 cell 19)"""
    payment_df = delivered_orders.merge(customers, on="customer_id").merge(
        payments, on="order_id"
    )
    return (
        payment_df.groupby(["customer_unique_id", "order_id"])[
            "payment_installments"
        ]
        .mean()
        .groupby("customer_unique_id")
        .mean()
        .rename("avg_installments")
    )


def compute_pct_credit_card(
    delivered_orders: pd.DataFrame,
    customers: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.Series:
    """Fraction of orders paid by credit card.  (NB2 cell 21)"""
    payment_df = delivered_orders.merge(customers, on="customer_id").merge(
        payments, on="order_id"
    )
    order_has_cc = payment_df.groupby(
        ["customer_unique_id", "order_id"]
    )["payment_type"].apply(lambda x: "credit_card" in x.values)
    return (
        order_has_cc.groupby("customer_unique_id")
        .mean()
        .rename("pct_credit_card")
    )


def compute_category_diversity(master_df: pd.DataFrame) -> pd.Series:
    """Number of distinct product categories purchased.  (NB2 cell 23)"""
    return (
        master_df.groupby("customer_unique_id")[
            "product_category_name_english"
        ]
        .nunique()
        .rename("category_diversity")
    )


def compute_is_positive_reviewer(
    delivered_orders: pd.DataFrame,
    customers: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.Series:
    """Binary: mean review score >= 4.  NaN → 0.  (NB2 cell 25)"""
    review_df = delivered_orders.merge(customers, on="customer_id").merge(
        reviews[["order_id", "review_score"]], on="order_id", how="left"
    )
    avg_review = review_df.groupby("customer_unique_id")["review_score"].mean()
    return (
        (avg_review >= 4).astype(int).fillna(0).rename("is_positive_reviewer")
    )


def compute_is_weekend_shopper(
    delivered_orders: pd.DataFrame, customers: pd.DataFrame
) -> pd.Series:
    """Binary: any purchase on Saturday (5) or Sunday (6).  (NB2 cell 27)"""
    weekend_df = delivered_orders.merge(customers, on="customer_id").copy()
    weekend_df["day_of_week"] = weekend_df[
        "order_purchase_timestamp"
    ].dt.dayofweek
    return (
        weekend_df.groupby("customer_unique_id")["day_of_week"]
        .apply(lambda x: ((x == 5) | (x == 6)).any())
        .astype(int)
        .rename("is_weekend_shopper")
    )


# ---------------------------------------------------------------------------
# Assembly and transforms  (NB2 cells 29-34)
# ---------------------------------------------------------------------------


def build_feature_matrix(raw_data: RawData) -> pd.DataFrame:
    """Compute all 9 raw features and assemble into one DataFrame."""
    master = raw_data.master_df
    delivered = raw_data.delivered_orders
    customers = raw_data.customers
    payments = raw_data.payments
    reviews = raw_data.reviews

    features = pd.concat(
        [
            compute_frequency(delivered, customers),
            compute_monetary_total(master),
            compute_monetary_avg_item(master),
            compute_avg_items_per_order(master),
            compute_avg_installments(delivered, customers, payments),
            compute_pct_credit_card(delivered, customers, payments),
            compute_category_diversity(master),
            compute_is_positive_reviewer(delivered, customers, reviews),
            compute_is_weekend_shopper(delivered, customers),
        ],
        axis=1,
    )

    # Drop customers with missing values (typically 1 customer)
    features = features.dropna()
    return features


def apply_transforms(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to skewed features. Returns 13-column DataFrame."""
    result = feature_matrix.copy()
    for col in LOG_TRANSFORM_FEATURES:
        result[f"{col}_log"] = np.log1p(result[col])
    return result


def run(raw_data: RawData, save: bool = True) -> FeatureData:
    """Pipeline entry point: raw data → feature matrix → transforms."""
    raw_features = build_feature_matrix(raw_data)
    complete = apply_transforms(raw_features)
    transformed = complete[CLUSTERING_FEATURES]

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        raw_features.to_csv(PROCESSED_DATA_DIR / "customer_features_raw.csv")

    return FeatureData(
        raw_features=raw_features,
        transformed_features=transformed,
        complete_features=complete,
    )
