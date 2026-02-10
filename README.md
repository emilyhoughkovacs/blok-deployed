# Blok Interview Prep: Persona-Based Behavioral Simulation

## Overview

This project demonstrates a foundational pipeline for behavioral simulation using customer personas derived from e-commerce transaction data. Built as interview preparation for [Blok](https://www.joinblok.co/), a behavioral simulation engine company, it bridges the gap between static customer segmentation and dynamic agent-based modeling.

**Core thesis**: Behavioral clustering provides the structural foundation for agent instantiation. By extracting latent behavioral patterns from real-world data, we can construct persona profiles that serve as initialization parameters for LLM-powered agents capable of simulating realistic customer decision-making.

## Motivation

Traditional customer segmentation yields descriptive insights (e.g., "high-value, infrequent buyers"), but doesn't capture the decision heuristics, contextual preferences, or cognitive patterns needed for predictive simulation. This project explores how to:

1. **Map behavioral data to actionable personas** — Identify clusters that represent meaningfully distinct behavioral archetypes, not just demographic groupings
2. **Translate static profiles into dynamic agents** — Use persona attributes (purchasing cadence, price sensitivity, category preferences) as behavioral priors for Claude-powered agents
3. **Validate simulation fidelity** — Test whether persona agents respond to product scenarios in ways consistent with the cluster behaviors they represent

This approach mirrors Blok's methodology: grounding synthetic user agents in real behavioral patterns to enable predictive, rather than purely retrospective, product testing.

## Dataset

**[Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)** (Kaggle)

100k orders from 2016–2018 with rich behavioral signals: purchase frequency, basket composition, payment preferences, review sentiment, and delivery experience. The granularity enables clustering by behavioral tendencies rather than demographics.

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/blok-deployed.git
cd blok-deployed
```

### 2. Download the data from Kaggle

Install the Kaggle CLI and configure your API credentials:

```bash
pip3 install kaggle
```

Go to https://www.kaggle.com/settings/account, scroll to "API", and click "Create New Token". This downloads `kaggle.json`. Move it into place:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Download and extract the dataset into the project's `data/raw/` directory:

```bash
kaggle datasets download -d olistbr/brazilian-ecommerce -p ./data/raw
unzip -o ./data/raw/brazilian-ecommerce.zip -d ./data/raw
```

### 3. Build the Docker image

```bash
docker compose build
```

This builds a container with Python 3.12, all dependencies, and the project source code. First build takes 2–3 minutes; subsequent rebuilds use cached layers and complete in seconds.

### 4. Configure your API key (for agent simulation)

Agent simulation calls the Claude API, which requires an Anthropic API key. Get one at https://console.anthropic.com/settings/keys, then create a `.env` file in the project root:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

Skip this step if you only plan to use mock mode (step 6).

### 5. Run the end-to-end pipeline

```bash
docker compose run --rm pipeline
```

This runs all 4 stages in sequence — load data, engineer features, cluster, generate personas — and saves outputs to `data/processed/`. The `--rm` flag auto-removes the container after it exits.

To pass additional flags, override the command:

```bash
docker compose run --rm pipeline python scripts/run_pipeline.py --evaluate-k
docker compose run --rm pipeline python scripts/run_pipeline.py --n-clusters 5
```

### 6. Run agent simulation

With mock responses (no API calls, good for testing the flow):

```bash
docker compose run --rm simulate-mock
```

With live Claude API responses (requires the `.env` file from step 4):

```bash
docker compose run --rm simulate
```

This runs 6 purchase scenarios across all 7 personas (42 simulations), validates consistency, and saves results to `notebooks/outputs/simulation/`.

### 7. Run tests

```bash
docker compose run --rm test
```

Runs 16 tests using synthetic data (no raw CSVs needed). Tests cover config validation, feature engineering, clustering, persona profiling, and agent mock responses. Smoke tests that require the raw data auto-skip if the CSVs aren't present.

### Local development (without Docker)

If you prefer running without Docker, create a virtual environment and install the package directly:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[all]"
```

Then run scripts directly:

```bash
python scripts/run_pipeline.py
python scripts/evaluate.py --mock
pytest tests/test_pipeline.py -v
```

**Interactive module testing** — import and run individual stages in a Python shell:

```python
from persona_clustering.data import loader
raw_data = loader.run(save=False)

from persona_clustering.features import engineering
features = engineering.run(raw_data, save=False)

from persona_clustering.models import clustering
result = clustering.run(features.transformed_features, save=False)

from persona_clustering.personas import profiler
personas = profiler.run(features.raw_features, result.labels, save=False)
```

Each module's `run()` function takes the output of the previous stage and returns a dataclass. Use `save=False` to keep it from writing files so you can inspect return values without side effects.

## Project Structure

```
├── src/persona_clustering/        # Main Python package
│   ├── config.py                  # Centralized paths, feature lists, hyperparameters
│   ├── data/loader.py             # Load and merge 7 Olist CSVs
│   ├── features/engineering.py    # Compute 9 behavioral features + log transforms
│   ├── models/clustering.py       # StandardScaler + KMeans (k=7)
│   └── personas/
│       ├── profiler.py            # Generate persona names, descriptions, system prompts
│       └── agent.py               # Claude agent wrapper + scenario simulation
├── scripts/
│   ├── run_pipeline.py            # CLI: data → features → clusters → personas
│   └── evaluate.py                # CLI: run agent simulation across scenarios
├── tests/
│   └── test_pipeline.py           # Unit + smoke tests (pytest)
├── notebooks/                     # Jupyter notebooks (exploration & visualization)
├── data/
│   ├── raw/                       # Olist CSV files (git-ignored)
│   └── processed/                 # Generated outputs (features, models, personas.json)
└── pyproject.toml                 # Package metadata and dependencies
```

## Workflow

The pipeline logic lives in the `persona_clustering` Python package. The notebooks remain available for interactive exploration and visualization, but the package is the authoritative implementation.

### Phase 1: Exploratory Data Analysis
[`01_eda_behavioral_clustering.ipynb`](notebooks/01_eda_behavioral_clustering.ipynb)

Explore the raw transaction data to understand distributions, identify behavioral signals, and assess data quality. Examines purchase patterns, payment methods, review behavior, and delivery outcomes.

### Phase 2: Feature Engineering
[`02_feature_engineering_clustering.ipynb`](notebooks/02_feature_engineering_clustering.ipynb) | `persona_clustering.features.engineering`

Transform raw transactions into customer-level behavioral features: purchase frequency, monetary value, basket size, installment usage, credit card preference, category diversity, review sentiment, and shopping timing.

### Phase 3: Clustering
[`03_clustering.ipynb`](notebooks/03_clustering.ipynb) | `persona_clustering.models.clustering`

Apply K-means clustering to identify distinct behavioral segments. Evaluate cluster quality using elbow method and silhouette scores. Final model: 7 clusters across 93,357 customers.

### Phase 4: Persona Profiling
[`04_persona_profiling.ipynb`](notebooks/04_persona_profiling.ipynb) | `persona_clustering.personas.profiler`

Characterize each cluster with representative statistics and z-scores. Generate natural language persona descriptions and decision heuristics. Output: structured persona profiles with LLM-ready system prompts.

### Phase 5: Agent Simulation
[`05_agent_simulation.ipynb`](notebooks/05_agent_simulation.ipynb) | `persona_clustering.personas.agent`

Instantiate Claude-powered agents from persona profiles. Run 6 product scenarios across all 7 personas (42 simulations). Validate that agent responses align with underlying behavioral profiles.

## Results

- **7 distinct personas** derived from real behavioral patterns (e.g., "High-Value Financing Shopper", "Critical Shopper", "Cash Customer")
- **100% validation alignment** — personas respond consistently with their cluster characteristics
- **Clear differentiation** — Critical Shopper rejected 6/6 scenarios; High-Value Financing Shopper accepted 5/6

## Scope

**Included**: EDA, feature engineering, clustering pipeline, persona profiling, Claude API agent instantiation, scenario simulation, validation framework

**Out of scope**: Production deployment, real-time backtesting, multi-agent interactions, calibration against historical conversion rates

## Why This Matters for Behavioral Simulation

Static personas are descriptive; agent-based personas are *generative*. By grounding LLM agents in empirically-derived behavioral clusters, we can predict responses to novel scenarios, explore counterfactuals before committing engineering resources, and identify edge-case behaviors that might respond unpredictably to standard interventions.

Customer clustering becomes the foundation for a simulation engine that can compress weeks of A/B testing into hours of agent-based experimentation.
