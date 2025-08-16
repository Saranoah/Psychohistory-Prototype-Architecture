# Real-Time Computational Psychohistory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Research Preview](https://img.shields.io/badge/status-research%20preview-orange.svg)]()

> *"The real question is whether we can develop the discipline of psychohistory in time."* - Isaac Asimov

## 🚀 Overview

The first practical implementation of **Real-Time Computational Psychohistory** - a system that monitors civilizational health using historical pattern recognition, live data integration, and AI-aware analysis to predict societal trends and potential instabilities.

Unlike previous attempts at historical prediction, this system:
- ⚡ **Monitors in real-time** using live data streams
- 🧠 **Accounts for AI influence** on human psychology and decision-making  
- 📊 **Integrates multiple dimensions** (economic, social, political, environmental, technological)
- 🎯 **Provides actionable insights** with specific recommendations
- 📈 **Learns from patterns** across civilizations and time periods


# Psychohistory Prototype Architecture (AI User Signals × Macro History)

**Goal:** Fuse daily AI-user interaction signals (ChatGPT, Claude, DeepSeek, Grok) with historical, economic, political, and environmental datasets to forecast civilizational stability and propose aligned interventions ("Zeroth-Law layer").

---

## 1) System Overview

**Ingest → Anonymize → Normalize → Feature-Fuse → Model (nowcast/forecast/ABM/causal) → Scenario Engine → Ethics Constraints → Policy Actions → Evaluation Loop**

* **Data Contracts** with each model/vendor (opt-in, anonymized, rate-limited, aggregated).
* **Privacy First**: irreversible hashing, DP noise, k-anonymity, and on-device pre-aggregation where possible.
* **Reproducibility**: versioned data and models, feature store, experiment tracking.

---

## 2) Data Sources & Schemas

### A. AI Interaction Streams (Event Schema)

* `event_id` (UUID), `timestamp` (UTC), `provider` ("openai"|"anthropic"|"xai"|"deepseek"),
* `session_id_hash`, `user_id_hash`, `georegion` (ISO/ coarse), `device_type`,
* `prompt_features` (tokens\_count, entropy, sentiment, topic labels, safety flags),
* `response_features` (latency\_ms, refusal\_rate, hallucination\_flag, coherence score),
* `engagement` (followups, edits, dwell\_time\_s),
* `privacy_bucket` (k-anon cohort id).

### B. Macro Signals

* **Economics**: CPI, unemployment, PMI, FX, commodities, equity indices.
* **Politics**: election calendars, regime type indices, protests (GDELT/ACLED), policy changes.
* **Social**: migration, education, health, social sentiment indices.
* **Environment**: temperature anomalies, drought index, disaster events.
* **Tech**: patent counts, OSS activity (GitHub), compute prices.

### C. Reference Taxonomies

* Regions, sectors, topics, risk categories (mapped to your `MetricCategory`).

---

## 3) Data Pipeline (Batch + Streaming)

* **Ingestion**: Kafka topics (e.g., `ai.interactions.raw`, `macro.market.raw`).
* **Stream Processing**: Flink/Spark Structured Streaming for real-time feature extraction (rolling windows, anomaly scores).
* **Batch ETL**: Airflow/Dagster orchestrates daily enrichments (join with macro datasets, backfills).
* **Storage**:

  * **Data Lake**: Parquet in S3/GS/Azure with Hive metastore (bronze/silver/gold).
  * **Warehouse**: BigQuery/Snowflake/Postgres for analytics.
  * **Graph DB**: Neo4j/ArangoDB to relate actors, events, institutions.
  * **Feature Store**: Feast/Bytewax for online/offline consistency.

**Privacy layer** (early in pipeline): salted hashing for IDs, DP noise addition to sensitive aggregates, cohorting.

---

## 4) Feature Engineering (Examples)

* **Interaction Micro-features**: prompt perplexity, novelty vs user’s cohort baseline, oscillation of topics, safety-trigger rate, model-switch behavior.

* **Provider Health**: latency SLO breaches, refusal/hallucination trend.

* **Cross-Modal Fusions**:

  * AI usage × market volatility (does uncertainty increase AI reliance?).
  * AI usage × protest frequency (information-seeking before events?).
  * OSS commits × patent filings × compute prices → tech momentum index.

* **Stability Feature Heads** mapped to: ECONOMIC, POLITICAL, SOCIAL, ENVIRONMENTAL, TECHNOLOGICAL.

---

## 5) Modeling Stack

### (a) Nowcasting

* Gradient boosted trees / LightGBM, plus Kalman filters for denoising real-time indices.

### (b) Time-Series Forecasting

* Probabilistic models (Prophet/NeuralProphet/Darts),
* Deep temporal models (Temporal Fusion Transformer),
* Regime switching (Markov switching VAR).

### (c) Causal Inference

* DoWhy/EconML for treatment effects (e.g., policy change impact on stability).
* Invariant risk minimization to generalize across regions.

### (d) Agent-Based Simulation (ABM)

* Micro-agents represent user cohorts & institutions; rules derived from causal graphs and historical analogues.
* Feed macro shocks (rate hikes, disasters) and policy levers; observe emergent stability metrics.

### (e) Ensemble & Scenario Engine

* Bayesian model averaging; scenario trees around Seldon-like crisis points.
* Monte Carlo over shocks ("black swans" with fat-tail priors).

---

## 6) Alignment & Ethics Layer ("Zeroth-Law Guardrails")

* **Utility function** combines stability, rights preservation, equity, and future option value.
* **Hard Constraints**: reject actions that reduce fundamental rights below threshold.
* **Counterfactual Audits**: only recommend interventions that pass fairness & harm tests across cohorts.
* **DP & Transparency**: publish privacy budgets and uncertainty intervals with every release.

---

## 7) Evaluation & Monitoring

* **Backtesting** on historical crises (e.g., 2008, Arab Spring, COVID-19) with rolling-origin evaluation.
* **Online Monitoring**: drift detection (PSI/K-S), calibration checks, SHAP drift.
* **Scorecards**: Brier score, CRPS, hit-rate on regime changes, policy outcome deltas.

---

## 8) MLOps

* **Experiment Tracking**: MLflow/Weights & Biases.
* **CI/CD**: GitHub Actions; unit tests for data contracts and feature logic.
* **Model Registry**: promote models behind feature flags; canary deployments.

---

## 9) Minimal Viable Prototype (MVP)

1. **Ingest** a simple stream: anonymized AI-interaction summaries + a macro dataset (e.g., CPI, VIX).
2. **Feature**: weekly aggregates per region/provider; compute a basic stability score.
3. **Model**: one nowcast head + 3-month forecast; uncertainty via quantile regression.
4. **Scenario**: inject two shocks (market crash; internet blackout) into ABM.
5. **Report**: dashboard with stability index, risk level, and protocol suggestions.

---

## 10) Example Tech Choices

* **Pipelines**: Kafka + Flink/Spark + Airflow/Dagster.
* **Storage**: S3/Parquet + BigQuery/Postgres + Neo4j + Feast.
* **Modeling**: Python (PyTorch/Sklearn), PyMC/NumPyro for probabilistic pieces, DoWhy/EconML.
* **Dashboards**: Streamlit/FastAPI backend, React (Next.js) frontend.

---

## 11) Interfaces & Code Skeletons

### A. Pluggable Collector Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseCollector(ABC):
    provider: str

    @abstractmethod
    async def fetch(self) -> Dict[str, Any]:
        ...

class ChatGPTCollector(BaseCollector):
    provider = "openai"
    async def fetch(self) -> Dict[str, Any]:
        # Call internal API/proxy; return **anonymized** event schema
        return {
            "provider": self.provider,
            "timestamp": datetime.utcnow().isoformat(),
            "prompt_features": {"tokens": 512, "entropy": 3.1},
            "response_features": {"latency_ms": 220},
        }
```

### B. Feature Registry (Feast-style)

```python
from dataclasses import dataclass
from typing import Callable, Dict

@dataclass
class FeatureDef:
    name: str
    fn: Callable[[Dict], float]

REGISTRY = [
    FeatureDef("prompt_entropy", lambda e: e["prompt_features"]["entropy"]),
    FeatureDef("latency_ms", lambda e: e["response_features"]["latency_ms"]),
]
```

### C. Stability Scorer (maps to your categories)

```python
def stability_score(features: Dict[str, float]) -> float:
    weights = {
        "ECONOMIC": 0.25,
        "POLITICAL": 0.25,
        "SOCIAL": 0.20,
        "TECHNOLOGICAL": 0.15,
        "ENVIRONMENTAL": 0.15,
    }
    # features already aggregated into these heads
    s = sum(features[k] * w for k, w in weights.items()) / sum(weights.values())
    return max(0.0, min(1.0, s))
```

### D. Scenario Engine Hook

```python
class ScenarioEngine:
    def __init__(self, base_state):
        self.base = base_state

    def shock(self, kind: str, magnitude: float):
        # mutate base in a reproducible way
        ...

    def run(self, horizon_months=12, sims=1000):
        # Monte Carlo over shocks + ABM interactions
        return {"p_collapse": 0.12, "p_renaissance": 0.33}
```

---

## 12) Governance & Compliance Checklist

* DPA/ToS-compliant data contracts; opt-in consent for interaction telemetry.
* Minimal data; delete raw prompts unless explicitly permitted; keep derived features only.
* Differential Privacy budgets & cohort-level reporting; red-team privacy attacks.
* External ethics board review for intervention recommendations.

---

## 13) Next Steps (Concrete)

1. Define the **event schema** and privacy budget; stub collectors for two providers.
2. Stand up **Kafka + Postgres** locally; implement stream aggregator to weekly region/provider features.
3. Train a first **nowcast/forecast** for a single stability index; ship a Streamlit dashboard.
4. Add a **scenario engine** with two shocks and an intervention recommender (your Foundation Protocols).
5. Backtest on 3 historical windows; iterate on causal features.

---

## Repository scaffold — initial files

I've added an initial repository scaffold and minimal starter files below. You can view, copy, or export these directly from this canvas. If you want, I can turn these into a downloadable zip or help push them to a GitHub repo next.

### Project tree (added)

```
psychohistory-monitor/
├── data/
│   ├── economic/
│   ├── cosmological/
│   └── social_media/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ROADMAP.md
│   └── ETHICS_CHARTER.md
├── models/
│   ├── cosmological_impact.py
│   ├── scientific_disruption.py
│   └── stability_core.py
├── visualization/
│   ├── cosmic_impact.py
│   ├── disruption_waves.py
│   └── timeline_projector.py
├── data_ingestion/
│   ├── ai_interfaces.py
│   ├── economic_feeds.py
│   └── simulated_feeds.py
├── tests/
│   ├── historical_validation/
│   └── future_projection/
├── .gitignore
├── LICENSE
├── requirements.txt
├── config.yaml
└── run.py
```

---

### `run.py` (bootstrap)

```python
"""run.py — Minimal entrypoint for the Psychohistory Monitor demo"""
from models.stability_core import EnhancedPsychohistoryEngine
from data_ingestion.simulated_feeds import generate_demo_data

if __name__ == "__main__":
    print("🚀 Initializing Psychohistory Monitor (demo)")
    engine = EnhancedPsychohistoryEngine()

    # Load sample data
    demo_data = generate_demo_data(days=180, cosmic_events=3, breakthroughs=5, crisis_points=2)

    # Run analysis
    stability = engine.calculate_stability_score(demo_data)
    patterns = engine.detect_patterns(demo_data)

    print(f"
🌍 Current Stability: {stability:.2f}/1.0")
    print("🔭 Detected Patterns:")
    for p in patterns:
        print(f" - {p['name']} (confidence: {p.get('significance',0):.0%})")

    print("
✅ Run `python -m visualization.timeline_projector` to launch the interactive dashboard (WIP)")
```

---

### `data_ingestion/simulated_feeds.py` (demo data generator)

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_cosmic_events(dates, count):
    events = []
    for _ in range(count):
        event_date = np.random.choice(dates)
        events.append({
            'date': pd.to_datetime(event_date),
            'type': np.random.choice(['dark_matter', 'exoplanet', 'gravitational_wave']),
            'significance': float(np.random.uniform(0.7, 0.95)),
            'philosophical_impact': float(np.random.uniform(0.5, 0.9))
        })
    return pd.DataFrame(events)


def generate_demo_data(days=90, cosmic_events=2, breakthroughs=3, crisis_points=1):
    dates = pd.date_range(end=datetime.utcnow(), periods=days)

    economic = pd.DataFrame({
        'date': dates,
        'inflation': np.clip(np.random.normal(0.02, 0.005, days), 0, 1),
        'unemployment': np.clip(np.random.uniform(0.038, 0.072, days), 0, 1),
        'consumer_confidence': np.clip(np.random.normal(50, 10, days), 0, 100)
    })

    cosmic = generate_cosmic_events(dates, cosmic_events)

    social = pd.DataFrame({
        'date': dates,
        'search_volume_crypto': np.random.poisson(20, days),
        'protest_mentions': np.random.poisson(2, days)
    })

    return {
        'economic': economic,
        'cosmological': cosmic,
        'social': social
    }
```

---

### `models/stability_core.py` (starter engine)

```python
from typing import Dict, Any, List

class EnhancedPsychohistoryEngine:
    """A minimal engine skeleton. Replace heuristics with models later."""
    def __init__(self):
        self.history = []

    def calculate_stability_score(self, demo_data: Dict[str, Any]) -> float:
        # Simple weighted heuristic for demo purposes
        econ = demo_data.get('economic')
        cosmic = demo_data.get('cosmological')
        social = demo_data.get('social')

        econ_score = 0.5
        if econ is not None and 'inflation' in econ:
            recent_infl = float(econ['inflation'].iloc[-1])
            econ_score = max(0.0, min(1.0, 1 - recent_infl))

        sci_shock = 0.0
        if isinstance(cosmic, (list, tuple)) or (hasattr(cosmic, 'shape') if cosmic is not None else False):
            sci_shock = float(getattr(cosmic, 'significance', 0.0)) if hasattr(cosmic, 'significance') else 0.0

        social_score = 0.5
        if social is not None and 'protest_mentions' in social:
            social_score = max(0.0, min(1.0, 1 - (social['protest_mentions'].iloc[-1] / 10)))

        # Weighted blend
        stability = 0.45 * econ_score + 0.35 * social_score + 0.20 * (1 - sci_shock)
        self.history.append({'ts': None, 'stability': stability})
        return float(max(0.0, min(1.0, stability)))

    def detect_patterns(self, demo_data: Dict[str, Any]) -> List[Dict]:
        patterns = []
        # Example heuristic detectors
        social = demo_data.get('social')
        if social is not None and 'protest_mentions' in social:
            recent = int(social['protest_mentions'].iloc[-7:].mean())
            if recent > 3:
                patterns.append({
                    'name': 'Rising Protest Mentions',
                    'significance': min(1.0, recent / 10.0),
                    'description': 'Sustained increase in protest-related mentions in social streams.'
                })
        return patterns
```

---

### `README.md` (starter)

````markdown
# Psychohistory Monitor (Demo)

Minimal demo of the Psychohistory Monitor. This repository contains simulated data generators, a starter stability engine, and visualization scaffolding to iterate quickly.

## Quick start

```bash
python run.py
````

## Goals

* Build a modular pipeline to fuse AI interaction signals with macro datasets and cosmological events.
* Provide a sandbox where researchers can plug in models and run reproducible experiments.

```

---

### `docs/ROADMAP.md` and `docs/ETHICS_CHARTER.md`
- Roadmap drafted (Phase 1..3) — see earlier canvas content for full roadmap.
- Ethics charter skeleton added; please review and expand in the canvas if you want specific clauses for data governance, red-team rules, and publication policies.

---

### `requirements.txt`
```

numpy
pandas
matplotlib
flask
seaborn
scikit-learn
python-dateutil

```

---

### `.gitignore`
```

**pycache**/
\*.pyc
.venv/
\*.db
.env

```

---

```
