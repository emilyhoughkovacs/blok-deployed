#!/usr/bin/env python
"""Run agent simulation and validation against generated personas."""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from persona_clustering.personas.agent import (
    PersonaSimulator,
    SCENARIOS,
    validate_persona_consistency,
    create_decision_heatmap,
)
from persona_clustering.config import PROCESSED_DATA_DIR, OUTPUTS_DIR


def main(
    personas_path: Path | None = None,
    mock_mode: bool = False,
    save: bool = True,
):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger(__name__)

    if personas_path is None:
        personas_path = PROCESSED_DATA_DIR / "personas.json"

    log.info(f"Loading personas from {personas_path}")
    simulator = PersonaSimulator(
        personas_path=personas_path,
        mock_mode=mock_mode,
    )
    simulator.load_personas()
    simulator.initialize_agents()
    log.info(
        f"  {len(simulator.agents)} agents initialized "
        f"({'mock' if mock_mode else 'live API'})"
    )

    log.info(
        f"Running {len(SCENARIOS)} scenarios x {len(simulator.agents)} personas..."
    )
    results = simulator.run_batch(SCENARIOS, structured=True)
    log.info(f"  Collected {len(results)} responses")

    # Validate
    validations = validate_persona_consistency(results)
    aligned = sum(1 for v in validations if v["aligned"])
    log.info(f"  Validation: {aligned}/{len(validations)} expectations aligned")

    for v in validations:
        status = "PASS" if v["aligned"] else "FAIL"
        log.info(f"    [{status}] {v['persona']} on {v['scenario']}")

    if save:
        output_dir = OUTPUTS_DIR / "simulation"
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_dir / "simulation_results.csv", index=False)
        create_decision_heatmap(
            results, output_path=output_dir / "decision_heatmap.png"
        )
        log.info(f"  Saved results to {output_dir}")

    return results, validations


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run persona agent simulation")
    parser.add_argument(
        "--personas", type=Path, default=None, help="Path to personas.json"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Use mock mode (no API calls)"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving results"
    )
    args = parser.parse_args()

    main(
        personas_path=args.personas,
        mock_mode=args.mock,
        save=not args.no_save,
    )
