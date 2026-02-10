"""Transform cluster centroids into behavioral personas with system prompts.

Extracted from NB4_persona_profiling.ipynb cells 7-29.  This is the most
logic-heavy module: it takes raw features + cluster labels and produces
named personas, behavioral profiles, decision heuristics, and Claude
system prompts ready for agent instantiation.
"""

import json
from dataclasses import dataclass

import pandas as pd

from persona_clustering.config import (
    RAW_FEATURES,
    Z_SCORE_THRESHOLD,
    PROCESSED_DATA_DIR,
)


@dataclass
class PersonaSet:
    """Container for the persona profiling stage outputs."""

    personas: dict
    cluster_centroids: pd.DataFrame
    cluster_zscores: pd.DataFrame
    population_stats: pd.DataFrame
    export_data: dict


# ---------------------------------------------------------------------------
# Statistical building blocks  (NB4 cells 7-9)
# ---------------------------------------------------------------------------


def compute_cluster_centroids(
    raw_features: pd.DataFrame, labels: pd.Series
) -> pd.DataFrame:
    """Mean of raw features per cluster.  (NB4 cell 7)"""
    df = raw_features[RAW_FEATURES].copy()
    df["cluster"] = labels
    return df.groupby("cluster")[RAW_FEATURES].mean()


def compute_population_stats(raw_features: pd.DataFrame) -> pd.DataFrame:
    """Mean / median / std across entire population.  (NB4 cell 8)"""
    return raw_features[RAW_FEATURES].agg(["mean", "median", "std"])


def compute_cluster_zscores(
    centroids: pd.DataFrame, pop_stats: pd.DataFrame
) -> pd.DataFrame:
    """How many std-devs each cluster centroid is from the pop mean.  (NB4 cell 9)"""
    return (centroids - pop_stats.loc["mean"]) / pop_stats.loc["std"]


# ---------------------------------------------------------------------------
# Descriptive labeling  (NB4 cells 14-15)
# ---------------------------------------------------------------------------


def describe_feature(
    feature_name: str,
    value: float,
    zscore: float,
    pop_stats: pd.DataFrame,
) -> str:
    """Human-readable description for a single feature value.  (NB4 cell 14)"""
    descriptions = {
        "frequency": {
            "metric": f"{value:.1f} orders",
            "behavior": (
                "repeat buyer"
                if value > 1.5
                else "one-time buyer" if value < 1.1 else "occasional repeat buyer"
            ),
        },
        "monetary_total": {
            "metric": f"R${value:.0f} lifetime",
            "behavior": (
                "high-value"
                if zscore > 1
                else "budget-conscious" if zscore < -1 else "moderate spender"
            ),
        },
        "monetary_avg_item": {
            "metric": f"R${value:.0f}/item",
            "behavior": (
                "premium buyer"
                if zscore > 1
                else "bargain hunter" if zscore < -1 else "mid-range buyer"
            ),
        },
        "avg_items_per_order": {
            "metric": f"{value:.1f} items/order",
            "behavior": (
                "bulk buyer"
                if value >= 3
                else "multi-item buyer" if value >= 2 else "single-item buyer"
            ),
        },
        "avg_installments": {
            "metric": f"{value:.1f} installments",
            "behavior": (
                "heavy financing"
                if value > 5
                else (
                    "cash/single payment"
                    if value < 1.5
                    else "moderate financing"
                )
            ),
        },
        "pct_credit_card": {
            "metric": f"{value * 100:.0f}% credit card",
            "behavior": (
                "credit card preference"
                if zscore > 0.5
                else (
                    "boleto/debit preference"
                    if zscore < -0.5
                    else "typical payment mix"
                )
            ),
        },
        "category_diversity": {
            "metric": f"{value:.1f} categories",
            "behavior": (
                "category explorer"
                if value > 1.5
                else "category focused" if value <= 1 else "slight explorer"
            ),
        },
        "is_positive_reviewer": {
            "metric": f"{value * 100:.0f}% positive",
            "behavior": (
                "satisfied customer"
                if value > 0.85
                else "critical reviewer" if value < 0.5 else "mixed satisfaction"
            ),
        },
        "is_weekend_shopper": {
            "metric": f"{value * 100:.0f}% weekend",
            "behavior": (
                "weekend shopper"
                if value > 0.5
                else "weekday shopper" if value < 0.15 else "any-day shopper"
            ),
        },
    }
    desc = descriptions.get(
        feature_name, {"metric": str(value), "behavior": "average"}
    )
    return f"{desc['metric']} ({desc['behavior']})"


def identify_distinguishing_features(
    cluster_id: int,
    zscores: pd.DataFrame,
    threshold: float = Z_SCORE_THRESHOLD,
) -> dict:
    """Features with |z-score| > threshold for a cluster.  (NB4 cell 15)"""
    cluster_z = zscores.loc[cluster_id]
    high = cluster_z[cluster_z > threshold].sort_values(ascending=False)
    low = cluster_z[cluster_z < -threshold].sort_values(ascending=True)
    return {
        "high": list(high.index),
        "low": list(low.index),
        "high_zscores": high.to_dict(),
        "low_zscores": low.to_dict(),
    }


# ---------------------------------------------------------------------------
# Cluster summary  (NB4 cell 17)
# ---------------------------------------------------------------------------


def generate_cluster_summary(
    cluster_id: int,
    centroids: pd.DataFrame,
    zscores: pd.DataFrame,
    pop_stats: pd.DataFrame,
    cluster_sizes: pd.Series,
) -> dict:
    """Assemble per-cluster statistics + descriptions into a summary dict."""
    centroid = centroids.loc[cluster_id]
    zscore = zscores.loc[cluster_id]
    size = cluster_sizes[cluster_id]
    pct = size / cluster_sizes.sum() * 100

    feature_descriptions = {}
    for feature in RAW_FEATURES:
        feature_descriptions[feature] = {
            "value": centroid[feature],
            "zscore": zscore[feature],
            "description": describe_feature(
                feature, centroid[feature], zscore[feature], pop_stats
            ),
        }

    distinguishing = identify_distinguishing_features(cluster_id, zscores)

    return {
        "cluster_id": cluster_id,
        "size": int(size),
        "percentage": pct,
        "features": feature_descriptions,
        "distinguishing_high": distinguishing["high"],
        "distinguishing_low": distinguishing["low"],
    }


# ---------------------------------------------------------------------------
# Persona generation  (NB4 cells 20-25)
# ---------------------------------------------------------------------------


def infer_decision_heuristics(summary: dict) -> list[str]:
    """Behavioral heuristics inferred from cluster stats.  (NB4 cell 20)"""
    heuristics: list[str] = []
    features = summary["features"]

    # --- Payment behavior (checked in order of specificity) ---
    installments = features["avg_installments"]["value"]
    cc_pct = features["pct_credit_card"]["value"]
    cc_zscore = features["pct_credit_card"]["zscore"]

    if installments > 5:
        heuristics.append(
            "I evaluate purchases by monthly payment size, not total cost. "
            "Spreading payments makes expensive items accessible."
        )
    elif cc_pct < 0.1:
        heuristics.append(
            "I avoid debt and prefer to pay upfront with boleto or debit. "
            "If I can't afford it now, I'll wait."
        )
    elif cc_zscore > 0.5 and installments < 2:
        heuristics.append(
            "I use credit cards for convenience and rewards, "
            "but pay off balances quickly."
        )
    elif cc_zscore > 0.5:
        heuristics.append(
            "Credit cards are my default payment method\u2014"
            "I appreciate the flexibility to pay over time."
        )

    # --- Spending behavior ---
    monetary_z = features["monetary_total"]["zscore"]
    avg_item_z = features["monetary_avg_item"]["zscore"]
    basket_value = features["avg_items_per_order"]["value"]

    if monetary_z > 1 and avg_item_z > 1:
        heuristics.append(
            "Quality matters more than price. "
            "I'm willing to pay premium for better products."
        )
    elif monetary_z < -0.5 and avg_item_z < -0.5:
        heuristics.append(
            "I'm price-conscious and actively seek deals. "
            "I compare prices before purchasing."
        )
    elif avg_item_z > 1 and basket_value < 2:
        heuristics.append(
            "I make deliberate, considered purchases. "
            "Each buy is a decision, not an impulse."
        )

    if basket_value >= 2:
        heuristics.append(
            "I prefer to bundle purchases together\u2014"
            "if I'm buying, I might as well get everything I need."
        )

    # --- Review behavior ---
    positive_pct = features["is_positive_reviewer"]["value"]

    if positive_pct > 0.85:
        heuristics.append(
            "I'm generally satisfied with my purchases "
            "and appreciate when things work as expected."
        )
    elif positive_pct < 0.5:
        heuristics.append(
            "I have high standards and will voice concerns "
            "when products don't meet expectations."
        )
    elif 0.5 <= positive_pct <= 0.75:
        heuristics.append(
            "I'm discerning but fair. "
            "I'll praise good experiences and critique poor ones."
        )

    # --- Shopping patterns ---
    weekend_pct = features["is_weekend_shopper"]["value"]
    frequency = features["frequency"]["value"]

    if weekend_pct > 0.5:
        heuristics.append(
            "I shop during leisure time, often browsing before buying."
        )
    elif weekend_pct < 0.15:
        heuristics.append(
            "I shop with purpose during the workweek, often for specific needs."
        )

    if frequency > 1.5:
        heuristics.append(
            "I'm comfortable with online shopping "
            "and return when I have a good experience."
        )
    elif frequency < 1.05:
        heuristics.append(
            "Online shopping is transactional for me\u2014"
            "I buy what I need and move on."
        )

    # --- Category behavior ---
    cat_diversity = features["category_diversity"]["value"]
    if cat_diversity > 1.5:
        heuristics.append(
            "I treat this marketplace as a one-stop shop for various needs."
        )
    elif cat_diversity <= 1:
        heuristics.append(
            "I come here for specific product types\u2014"
            "I know what I'm looking for."
        )

    return heuristics


def generate_persona_name(summary: dict) -> str:
    """Auto-generate a descriptive persona name.  (NB4 cell 21)"""
    features = summary["features"]
    high = summary["distinguishing_high"]
    low = summary["distinguishing_low"]

    name_parts: list[str] = []

    monetary_z = features["monetary_total"]["zscore"]
    if monetary_z > 1.5:
        name_parts.append("Premium")
    elif monetary_z > 0.75:
        name_parts.append("High-Value")
    elif monetary_z < -0.75:
        name_parts.append("Budget")

    if "avg_installments" in high:
        name_parts.append("Financing")
    elif "pct_credit_card" in low:
        name_parts.append("Cash")

    if "is_positive_reviewer" in high:
        name_parts.append("Satisfied")
    elif "is_positive_reviewer" in low:
        name_parts.append("Critical")

    if "is_weekend_shopper" in high:
        name_parts.append("Weekend")
    elif "frequency" in high:
        name_parts.append("Loyal")

    if "avg_items_per_order" in high:
        name_parts.append("Bulk")
    if "category_diversity" in high:
        name_parts.append("Explorer")

    if not name_parts:
        name_parts.append("Mainstream")

    suffixes = ["Shopper", "Buyer", "Customer"]
    return " ".join(name_parts[:2]) + " " + suffixes[summary["cluster_id"] % 3]


def generate_persona_description(summary: dict) -> dict:
    """Full persona dict with profile, heuristics, stats.  (NB4 cell 22)"""
    cluster_id = summary["cluster_id"]
    features = summary["features"]

    persona_name = generate_persona_name(summary)
    heuristics = infer_decision_heuristics(summary)

    # --- Behavioral profile sentences ---
    behavioral_profile: list[str] = []

    freq = features["frequency"]["value"]
    monetary = features["monetary_total"]["value"]
    avg_item = features["monetary_avg_item"]["value"]

    if freq > 1.5:
        behavioral_profile.append(
            f"Repeat customer with {freq:.1f} orders on average"
        )
    else:
        behavioral_profile.append("Typically makes a single purchase")

    behavioral_profile.append(f"Average lifetime spend of R${monetary:.0f}")
    behavioral_profile.append(f"Typical item price around R${avg_item:.0f}")

    basket = features["avg_items_per_order"]["value"]
    if basket >= 3:
        behavioral_profile.append(
            f"Bulk buyer with {basket:.1f} items per order on average"
        )
    elif basket >= 2:
        behavioral_profile.append(
            f"Buys multiple items per order ({basket:.1f} items on average)"
        )
    else:
        behavioral_profile.append("Usually buys one item per order")

    installments = features["avg_installments"]["value"]
    cc_pct = features["pct_credit_card"]["value"]
    cc_zscore = features["pct_credit_card"]["zscore"]

    if cc_zscore > 0.5:
        payment_desc = "Predominantly uses credit card"
    elif cc_zscore < -0.5:
        payment_desc = "Prefers boleto/debit over credit"
    else:
        payment_desc = f"Uses credit card for {cc_pct * 100:.0f}% of purchases"

    if installments > 5:
        payment_desc += f", typically in {installments:.0f} installments"
    elif installments < 1.5:
        payment_desc += ", usually paying in full"
    else:
        payment_desc += f", averaging {installments:.1f} installments"
    behavioral_profile.append(payment_desc)

    cat_div = features["category_diversity"]["value"]
    if cat_div > 1.5:
        behavioral_profile.append(
            f"Explores multiple product categories ({cat_div:.1f} on average)"
        )
    else:
        behavioral_profile.append("Focused on specific product categories")

    positive_pct = features["is_positive_reviewer"]["value"]
    if positive_pct > 0.85:
        behavioral_profile.append(
            "Highly satisfied\u2014reviews are consistently positive"
        )
    elif positive_pct > 0.7:
        behavioral_profile.append(
            "Generally satisfied, with occasional concerns"
        )
    elif positive_pct > 0.5:
        behavioral_profile.append(
            "Mixed satisfaction\u2014reviews reflect both positive and negative experiences"
        )
    else:
        behavioral_profile.append(
            "Often critical in reviews\u2014holds products to high standards"
        )

    weekend_pct = features["is_weekend_shopper"]["value"]
    if weekend_pct > 0.5:
        behavioral_profile.append("Shops primarily on weekends")
    elif weekend_pct < 0.15:
        behavioral_profile.append("Shops primarily on weekdays")
    else:
        behavioral_profile.append("No strong weekday/weekend preference")

    return {
        "cluster_id": cluster_id,
        "persona_name": persona_name,
        "behavioral_profile": behavioral_profile,
        "decision_heuristics": heuristics,
        "raw_statistics": {
            feature: {
                "value": features[feature]["value"],
                "zscore": features[feature]["zscore"],
            }
            for feature in RAW_FEATURES
        },
        "size": summary["size"],
        "percentage": summary["percentage"],
    }


def generate_agent_system_prompt(persona: dict) -> str:
    """Claude system prompt for a persona agent.  (NB4 cell 25)"""
    stats = persona["raw_statistics"]

    def _fmt_z(z: float) -> str:
        sign = "+" if z > 0 else ""
        return f"{sign}{z:.1f}\u03c3"

    prompt = f'''You are simulating a customer from the "{persona['persona_name']}" behavioral segment.

## Context
- Brazilian e-commerce customer (Olist marketplace, 2016-2018)
- This persona represents {persona['percentage']:.1f}% of the customer base ({persona['size']:,} customers)

## Key Statistics (with difference from population mean)

| Metric | Value | vs. Population |
|--------|-------|----------------|
| Purchase Frequency | {stats['frequency']['value']:.1f} orders | {_fmt_z(stats['frequency']['zscore'])} |
| Lifetime Spend | R${stats['monetary_total']['value']:.0f} | {_fmt_z(stats['monetary_total']['zscore'])} |
| Avg Item Price | R${stats['monetary_avg_item']['value']:.0f} | {_fmt_z(stats['monetary_avg_item']['zscore'])} |
| Items per Order | {stats['avg_items_per_order']['value']:.1f} | {_fmt_z(stats['avg_items_per_order']['zscore'])} |
| Avg Installments | {stats['avg_installments']['value']:.1f} | {_fmt_z(stats['avg_installments']['zscore'])} |
| Credit Card Usage | {stats['pct_credit_card']['value']*100:.0f}% | {_fmt_z(stats['pct_credit_card']['zscore'])} |
| Category Diversity | {stats['category_diversity']['value']:.1f} categories | {_fmt_z(stats['category_diversity']['zscore'])} |
| Positive Reviews | {stats['is_positive_reviewer']['value']*100:.0f}% | {_fmt_z(stats['is_positive_reviewer']['zscore'])} |
| Weekend Shopping | {stats['is_weekend_shopper']['value']*100:.0f}% | {_fmt_z(stats['is_weekend_shopper']['zscore'])} |

Note: \u03c3 (sigma) indicates standard deviations from population mean. Positive values are above average, negative values are below average.

## Behavioral Summary
- Shopping timing: {"weekend-oriented" if stats['is_weekend_shopper']['value'] > 0.5 else "weekday-oriented" if stats['is_weekend_shopper']['value'] < 0.2 else "no strong day preference"}
- Payment style: {"credit card with installment financing" if stats['pct_credit_card']['value'] > 0.5 and stats['avg_installments']['value'] > 3 else "credit card, pays quickly" if stats['pct_credit_card']['value'] > 0.5 else "prefers boleto/debit (pays upfront)"}
- Basket behavior: {"bulk buyer (multiple items)" if stats['avg_items_per_order']['value'] >= 2 else "single-item purchases"}

## Decision Heuristics
{chr(10).join("- " + h for h in persona['decision_heuristics'])}

## Instructions
When presented with product scenarios, purchasing decisions, or marketplace situations:

1. Respond as this customer persona would, based on the statistics and heuristics above
2. Your preferences should reflect:
   - The economic constraints implied by your spending patterns (note where you are vs. population)
   - The risk tolerance implied by your payment preferences
   - The satisfaction threshold implied by your review behavior
3. Stay in character throughout the conversation
4. When making decisions, briefly explain your reasoning in a way consistent with your persona's profile

Do not break character or acknowledge that you are an AI simulating a customer.

## Response Format
Always structure your response with these sections:

DECISION: [Yes/No/Maybe]

REASONING: [2-3 sentences explaining your decision from this persona's perspective]

KEY FACTORS: [Bullet points listing the most important factors that influenced your decision]
'''
    return prompt.strip()


# ---------------------------------------------------------------------------
# Pipeline entry point  (NB4 cells 22-27)
# ---------------------------------------------------------------------------


def run(
    raw_features: pd.DataFrame,
    labels: pd.Series,
    save: bool = True,
) -> PersonaSet:
    """Pipeline entry point: features + labels → PersonaSet with export data."""
    centroids = compute_cluster_centroids(raw_features, labels)
    pop_stats = compute_population_stats(raw_features)
    zscores = compute_cluster_zscores(centroids, pop_stats)
    cluster_sizes = labels.value_counts().sort_index()

    # Build summaries then persona descriptions
    personas: dict = {}
    for cluster_id in centroids.index:
        summary = generate_cluster_summary(
            cluster_id, centroids, zscores, pop_stats, cluster_sizes
        )
        persona = generate_persona_description(summary)
        persona["agent_system_prompt"] = generate_agent_system_prompt(persona)
        personas[cluster_id] = persona

    # Build export dict matching existing personas.json schema
    export_data = {
        "metadata": {
            "n_clusters": len(personas),
            "total_customers": int(cluster_sizes.sum()),
            "features_used": RAW_FEATURES,
            "generated_from": "persona_clustering.personas.profiler",
        },
        "population_statistics": {
            "mean": pop_stats.loc["mean"].to_dict(),
            "median": pop_stats.loc["median"].to_dict(),
            "std": pop_stats.loc["std"].to_dict(),
        },
        "personas": {
            int(cid): {
                "persona_name": p["persona_name"],
                "size": p["size"],
                "percentage": p["percentage"],
                "behavioral_profile": p["behavioral_profile"],
                "decision_heuristics": p["decision_heuristics"],
                "raw_statistics": {
                    k: {"value": float(v["value"]), "zscore": float(v["zscore"])}
                    for k, v in p["raw_statistics"].items()
                },
                "agent_system_prompt": p["agent_system_prompt"],
            }
            for cid, p in personas.items()
        },
    }

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_DATA_DIR / "personas.json", "w") as f:
            json.dump(export_data, f, indent=2)

    return PersonaSet(
        personas=personas,
        cluster_centroids=centroids,
        cluster_zscores=zscores,
        population_stats=pop_stats,
        export_data=export_data,
    )
