"""Persona-based behavioral simulation agents.

Relocated from src/agents.py with additions from NB5_agent_simulation.ipynb:
- SCENARIOS constant (NB5 cell 9)
- validate_persona_consistency() (NB5 cell 23)
- create_decision_heatmap() (NB5 cell 21)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from persona_clustering.config import DEFAULT_MODEL, PROCESSED_DATA_DIR

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Scenarios for agent simulation  (NB5 cell 9)
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "financing_offer",
        "text": (
            "You're looking at a smartphone priced at R$1,200. "
            "The seller offers 10x installments of R$120 with no interest on credit card. "
            "Alternatively, you can pay R$1,080 upfront via boleto (10% discount). "
            "The phone has good reviews (4.2 stars) and ships in 3 days. "
            "Would you buy this phone? Which payment option would you choose?"
        ),
    },
    {
        "name": "flash_sale",
        "text": (
            "FLASH SALE: A kitchen blender you've been eyeing is now 40% off! "
            "Original price: R$250 → Sale price: R$150. "
            "The catch: Sale ends in 2 hours. You weren't planning to buy today. "
            "Reviews are positive (4.5 stars), and you've needed a blender for a while. "
            "Do you buy it now, or wait and risk missing the deal?"
        ),
    },
    {
        "name": "bundle_deal",
        "text": (
            "You need to buy replacement ink cartridges for your printer. "
            "Single cartridge: R$45 each. Bundle of 3: R$115 (save 15%). "
            "You definitely need 1 now, might need the others in 3-6 months. "
            "Would you buy the single cartridge or the bundle?"
        ),
    },
    {
        "name": "new_category",
        "text": (
            "You've never bought fitness equipment online before, but you see an ad: "
            "Adjustable dumbbells set for R$280 with 4.3 star reviews. "
            "You've been thinking about working out at home. "
            "This marketplace has always worked well for your electronics purchases. "
            "Would you try buying fitness equipment from this platform?"
        ),
    },
    {
        "name": "mixed_reviews",
        "text": (
            "You found the exact wireless earbuds you want at R$180. "
            "However, the reviews are mixed: 3.5 stars overall. "
            "Positive reviews praise sound quality. Negative reviews mention "
            "Bluetooth connectivity issues. "
            "The seller has a 30-day return policy. "
            "Would you take the risk on these earbuds?"
        ),
    },
    {
        "name": "weekend_promo",
        "text": (
            "It's Tuesday, and you see a promotion: "
            "'WEEKEND SPECIAL: 20% off all home decor items, Saturday-Sunday only!' "
            "There's a decorative lamp you like for R$120 (would be R$96 with discount). "
            "You could buy now at full price and get it by Thursday, "
            "or wait for the weekend, buy at discount, and receive it next week. "
            "What would you do?"
        ),
    },
]


# ---------------------------------------------------------------------------
# PersonaAgent  (from src/agents.py)
# ---------------------------------------------------------------------------


@dataclass
class PersonaAgent:
    """Wraps the Claude API with a persona's system prompt."""

    cluster_id: int
    persona_name: str
    system_prompt: str
    client: Any = None
    mock_mode: bool = False
    model: str = DEFAULT_MODEL

    @classmethod
    def from_persona_data(
        cls,
        persona_data: dict,
        client: Any = None,
        mock_mode: bool = False,
        model: str = DEFAULT_MODEL,
    ) -> PersonaAgent:
        """Create a PersonaAgent from persona dictionary."""
        return cls(
            cluster_id=persona_data.get("cluster_id", -1),
            persona_name=persona_data["persona_name"],
            system_prompt=persona_data["agent_system_prompt"],
            client=client,
            mock_mode=mock_mode,
            model=model,
        )

    def respond(self, scenario: str, max_tokens: int = 500) -> str:
        """Generate a response to a product/purchase scenario."""
        if self.mock_mode:
            return self._mock_response(scenario)

        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        if self.client is None:
            raise ValueError(
                "No Anthropic client provided. "
                "Either pass a client or use mock_mode=True"
            )

        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": scenario}],
        )
        return message.content[0].text

    def respond_with_decision(
        self, scenario: str, max_tokens: int = 500
    ) -> dict:
        """Generate a structured response with decision and reasoning."""
        structured_prompt = f"""{scenario}

Please respond with:
1. DECISION: [Yes/No/Maybe] - Would you make this purchase?
2. REASONING: Brief explanation of your decision (2-3 sentences)
3. KEY FACTORS: What were the most important factors in your decision?"""

        raw_response = self.respond(structured_prompt, max_tokens)
        decision = self._extract_decision(raw_response)

        return {
            "persona_name": self.persona_name,
            "cluster_id": self.cluster_id,
            "decision": decision,
            "raw_response": raw_response,
        }

    def _extract_decision(self, response: str) -> str:
        """Extract decision from structured response."""
        import re

        response_lower = response.lower()

        decision_pattern = r"\*{0,2}decision\*{0,2}[:\s]+\[?(\w+)\]?"
        match = re.search(decision_pattern, response_lower)

        if match:
            decision = match.group(1)
            if decision in ("yes", "y"):
                return "Yes"
            elif decision in ("no", "n"):
                return "No"
            elif decision in ("maybe", "uncertain", "unsure"):
                return "Maybe"

        first_part = response_lower[:200]
        if (
            "i would buy" in first_part
            or "i'll take" in first_part
            or "yes," in first_part
        ):
            return "Yes"
        elif (
            "i would not" in first_part
            or "i wouldn't" in first_part
            or "no," in first_part
        ):
            return "No"

        return "Unclear"

    def _mock_response(self, scenario: str) -> str:
        """Generate a mock response for testing without API."""
        mock_templates = {
            "Mainstream Shopper": (
                "As a typical weekday shopper, I'd consider this purchase carefully. "
                "I usually buy what I need and move on. Given this scenario, I'd likely "
                "proceed if it meets my specific need and the price is reasonable."
            ),
            "Weekend Buyer": (
                "I typically browse on weekends when I have time. This seems interesting, "
                "but I'd want to think it over during my weekend shopping time."
            ),
            "Cash Customer": (
                "I prefer to pay upfront with boleto. If this requires installments or "
                "credit, I'd hesitate. I don't like carrying debt for purchases."
            ),
            "High-Value Financing Shopper": (
                "I'm comfortable with larger purchases when I can spread payments. "
                "If 10x installments are available, the monthly cost matters more "
                "than total price."
            ),
            "Bulk Buyer": (
                "I prefer to bundle purchases together. If there's a deal for buying "
                "multiple, I'd be more interested. Single items feel less efficient to me."
            ),
            "Loyal Explorer Customer": (
                "I'm always open to trying new categories. As a repeat customer, I trust "
                "this marketplace and would consider exploring this option."
            ),
            "Critical Shopper": (
                "I have high standards. Before deciding, I'd want to see the reviews "
                "carefully. If there are quality concerns, I'd pass regardless of the price."
            ),
        }

        base_response = mock_templates.get(
            self.persona_name,
            f"[Mock response for {self.persona_name}] Considering the scenario...",
        )

        return (
            f"[MOCK MODE] {base_response}\n\n"
            f"Scenario received: {scenario[:100]}..."
        )


# ---------------------------------------------------------------------------
# PersonaSimulator  (from src/agents.py, path default updated)
# ---------------------------------------------------------------------------


class PersonaSimulator:
    """Runs scenarios across all personas and collects responses."""

    def __init__(
        self,
        personas_path: str | Path | None = None,
        mock_mode: bool = False,
        model: str = DEFAULT_MODEL,
    ):
        if personas_path is None:
            personas_path = PROCESSED_DATA_DIR / "personas.json"

        self.personas_path = Path(personas_path)
        self.mock_mode = mock_mode
        self.model = model
        self.personas_data: dict = {}
        self.agents: dict[int, PersonaAgent] = {}
        self._client = None

    def load_personas(self, data: dict | None = None) -> dict:
        """Load personas from JSON file or in-memory dict."""
        if data is not None:
            self.personas_data = data
        else:
            with open(self.personas_path, "r") as f:
                self.personas_data = json.load(f)
        return self.personas_data

    def _get_client(self):
        """Get or create Anthropic client."""
        if self.mock_mode:
            return None

        if self._client is None:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "anthropic package not installed. "
                    "Run: pip install anthropic"
                )
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in environment. "
                    "Set it or use mock_mode=True"
                )
            self._client = anthropic.Anthropic(api_key=api_key)

        return self._client

    def initialize_agents(self) -> dict[int, PersonaAgent]:
        """Create PersonaAgent instances for all personas."""
        if not self.personas_data:
            self.load_personas()

        client = self._get_client()

        for cluster_id_str, persona_data in self.personas_data[
            "personas"
        ].items():
            cluster_id = int(cluster_id_str)
            persona_data["cluster_id"] = cluster_id
            self.agents[cluster_id] = PersonaAgent.from_persona_data(
                persona_data,
                client=client,
                mock_mode=self.mock_mode,
                model=self.model,
            )

        return self.agents

    def run_scenario(
        self, scenario: str, structured: bool = True
    ) -> pd.DataFrame:
        """Run a single scenario across all personas."""
        if not self.agents:
            self.initialize_agents()

        results = []
        for cluster_id, agent in sorted(self.agents.items()):
            if structured:
                result = agent.respond_with_decision(scenario)
                results.append(
                    {
                        "cluster_id": cluster_id,
                        "persona_name": result["persona_name"],
                        "decision": result["decision"],
                        "response": result["raw_response"],
                    }
                )
            else:
                response = agent.respond(scenario)
                results.append(
                    {
                        "cluster_id": cluster_id,
                        "persona_name": agent.persona_name,
                        "decision": None,
                        "response": response,
                    }
                )

        return pd.DataFrame(results)

    def run_batch(
        self, scenarios: list[dict], structured: bool = True
    ) -> pd.DataFrame:
        """Run multiple scenarios across all personas."""
        all_results = []
        for scenario in scenarios:
            scenario_name = scenario.get("name", "unnamed")
            df = self.run_scenario(scenario["text"], structured=structured)
            df["scenario_name"] = scenario_name
            all_results.append(df)

        return pd.concat(all_results, ignore_index=True)

    def get_persona_summary(self) -> pd.DataFrame:
        """Get a summary of all loaded personas."""
        if not self.personas_data:
            self.load_personas()

        summaries = []
        for cluster_id_str, persona in self.personas_data["personas"].items():
            summaries.append(
                {
                    "cluster_id": int(cluster_id_str),
                    "persona_name": persona["persona_name"],
                    "size": persona["size"],
                    "percentage": f"{persona['percentage']:.1f}%",
                }
            )

        return pd.DataFrame(summaries).sort_values("cluster_id")


# ---------------------------------------------------------------------------
# Validation and visualization  (NB5 cells 21, 23)
# ---------------------------------------------------------------------------


def validate_persona_consistency(results_df: pd.DataFrame) -> list[dict]:
    """Check that personas respond as expected for key scenarios.  (NB5 cell 23)"""
    expectations = [
        {
            "persona": "Cash Customer",
            "scenario": "financing_offer",
            "expected": "Should prefer boleto/upfront payment over installments",
            "check_keywords": [
                "boleto",
                "upfront",
                "discount",
                "cash",
                "pay now",
            ],
        },
        {
            "persona": "High-Value Financing Shopper",
            "scenario": "financing_offer",
            "expected": "Should prefer installment option, focus on monthly cost",
            "check_keywords": [
                "installment",
                "monthly",
                "spread",
                "payment plan",
            ],
        },
        {
            "persona": "Bulk Buyer",
            "scenario": "bundle_deal",
            "expected": "Should prefer the bundle over single item",
            "check_keywords": ["bundle", "three", "save", "stock up"],
        },
        {
            "persona": "Critical Shopper",
            "scenario": "mixed_reviews",
            "expected": "Should be hesitant due to negative reviews",
            "check_keywords": [
                "concern",
                "risk",
                "issue",
                "hesitant",
                "wait",
                "no",
            ],
        },
        {
            "persona": "Loyal Explorer Customer",
            "scenario": "new_category",
            "expected": "Should be open to trying new category",
            "check_keywords": ["try", "open", "trust", "explore", "yes"],
        },
        {
            "persona": "Weekend Buyer",
            "scenario": "weekend_promo",
            "expected": "Should prefer waiting for weekend deal",
            "check_keywords": [
                "wait",
                "weekend",
                "discount",
                "saturday",
                "sunday",
            ],
        },
    ]

    validations = []
    for exp in expectations:
        row = results_df[
            (results_df["persona_name"] == exp["persona"])
            & (results_df["scenario_name"] == exp["scenario"])
        ]
        if row.empty:
            continue

        response = row.iloc[0]["response"].lower()
        decision = row.iloc[0]["decision"]
        keywords_found = [kw for kw in exp["check_keywords"] if kw in response]

        validations.append(
            {
                "persona": exp["persona"],
                "scenario": exp["scenario"],
                "expectation": exp["expected"],
                "decision": decision,
                "keywords_found": keywords_found,
                "aligned": len(keywords_found) > 0,
            }
        )

    return validations


def create_decision_heatmap(
    results_df: pd.DataFrame, output_path: Path | None = None
) -> None:
    """Visualize persona decisions as a heatmap.  (NB5 cell 21)"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    decision_pivot = results_df.pivot_table(
        index="persona_name",
        columns="scenario_name",
        values="decision",
        aggfunc="first",
    )

    decision_map = {"Yes": 1, "Maybe": 0.5, "No": 0, "Unclear": 0.25}
    decision_numeric = decision_pivot.replace(decision_map)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        decision_numeric,
        annot=decision_pivot.values,
        fmt="",
        cmap="RdYlGn",
        center=0.5,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Decision (Yes=1, No=0)"},
    )
    ax.set_title("Persona Decisions Across Scenarios", fontsize=14)
    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Persona", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
