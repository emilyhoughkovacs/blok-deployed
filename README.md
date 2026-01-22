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

### Setup Instructions

1. **Install Kaggle CLI** (in your virtual environment):
   ```bash
   pip3 install kaggle
   ```

2. **Configure Kaggle API credentials**:
   - Go to https://www.kaggle.com/settings/account
   - Scroll to "API" section and click "Create New Token"
   - This downloads `kaggle.json` with your credentials
   - Move it to the correct location:
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Download and extract the dataset**:
   ```bash
   kaggle datasets download -d olistbr/brazilian-ecommerce -p ./data/raw
   unzip -o ./data/raw/brazilian-ecommerce.zip -d ./data/raw
   ```

## High-Level Workflow

### Phase 1: Behavioral Clustering
Extract behavioral patterns from transaction data and identify distinct customer archetypes through unsupervised clustering.

### Phase 2: Persona Profiling
Characterize each cluster with representative statistics and behavioral tendencies, then generate natural language persona descriptions.

### Phase 3: Agent Instantiation
Build persona agents using the Claude API, initializing each with its persona profile to guide decision-making behavior.

### Phase 4: Scenario-Based Simulation
Test persona agents against product scenarios and compare responses across behavioral archetypes.

## Scope

**Included**: Clustering pipeline, persona profiling, Claude API agent instantiation, scenario simulation
**Out of scope**: Production deployment, real-time backtesting, multi-agent interactions

## Why This Matters for Behavioral Simulation

Static personas are descriptive; agent-based personas are *generative*. By grounding LLM agents in empirically-derived behavioral clusters, we can predict responses to novel scenarios, explore counterfactuals before committing engineering resources, and identify edge-case behaviors that might respond unpredictably to standard interventions.

Customer clustering becomes the foundation for a simulation engine that can compress weeks of A/B testing into hours of agent-based experimentation.

---

**Status**: Repository initialized. Implementation to follow.
