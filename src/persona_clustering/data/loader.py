"""Load raw Olist CSVs and build the master order-customer-item table.

Extracted from NB2_feature_engineering.ipynb cells 4-8.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from persona_clustering.config import (
    RAW_DATA_DIR,
    RAW_DATASETS,
    ORDERS_DATE_COLS,
    REVIEWS_DATE_COLS,
)


@dataclass
class RawData:
    """Container for all raw DataFrames needed downstream."""

    master_df: pd.DataFrame
    delivered_orders: pd.DataFrame
    customers: pd.DataFrame
    payments: pd.DataFrame
    reviews: pd.DataFrame
    products: pd.DataFrame


def load_raw_datasets(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the 7 Olist CSVs with date parsing where applicable."""
    data_dir = data_dir or RAW_DATA_DIR

    customers = pd.read_csv(data_dir / RAW_DATASETS["customers"])

    orders = pd.read_csv(
        data_dir / RAW_DATASETS["orders"],
        parse_dates=ORDERS_DATE_COLS,
    )

    order_items = pd.read_csv(
        data_dir / RAW_DATASETS["order_items"],
        parse_dates=["shipping_limit_date"],
    )

    payments = pd.read_csv(data_dir / RAW_DATASETS["payments"])

    reviews = pd.read_csv(
        data_dir / RAW_DATASETS["reviews"],
        parse_dates=REVIEWS_DATE_COLS,
    )

    products = pd.read_csv(data_dir / RAW_DATASETS["products"])

    category_translation = pd.read_csv(
        data_dir / "product_category_name_translation.csv"
    )

    # Merge products with English category names
    products = products.merge(
        category_translation, on="product_category_name", how="left"
    )

    return {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
        "reviews": reviews,
        "products": products,
    }


def build_master_table(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter to delivered orders and merge into a single master table.

    Returns (master_df, delivered_orders).
    """
    delivered_orders = orders[orders["order_status"] == "delivered"].copy()

    master_df = (
        delivered_orders.merge(customers, on="customer_id")
        .merge(order_items, on="order_id")
        .merge(
            products[["product_id", "product_category_name_english"]],
            on="product_id",
            how="left",
        )
    )

    return master_df, delivered_orders


def run(data_dir: Path | None = None, save: bool = True) -> RawData:
    """Pipeline entry point: load CSVs → build master table → return RawData."""
    datasets = load_raw_datasets(data_dir)

    master_df, delivered_orders = build_master_table(
        datasets["customers"],
        datasets["orders"],
        datasets["order_items"],
        datasets["products"],
    )

    return RawData(
        master_df=master_df,
        delivered_orders=delivered_orders,
        customers=datasets["customers"],
        payments=datasets["payments"],
        reviews=datasets["reviews"],
        products=datasets["products"],
    )
