import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field
import logging
import matplotlib.pyplot as plt
import unittest
from copy import copy
from collections import OrderedDict

# Set up configurable logging
def setup_logging(log_level: str = "INFO"):
    """Configure logging with specified level."""
    logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== MODULE IMPLEMENTATIONS ====================

class MetricCategory(Enum):
    ECONOMIC = "economic"
    POLITICAL = "political"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"
    TECHNOLOGICAL = "technological"

@dataclass
class HistoricalPattern:
    name: str
    preconditions: Dict[str, Tuple[float, float]]  # Metric key: (min_val, max_val)
    outcome: str
    confidence_threshold: float = 0.7
    severity: float = 0.5
    timeframe: str = "medium_term"

    def get_metric_weight(self, metric_key: str) -> float:
        """Return weight for a metric in pattern matching.

        Args:
            metric_key (str): Metric identifier (e.g., 'ECONOMIC_wealth_inequality')

        Returns:
            float: Weight for the metric, defaulting to equal distribution if not specified

        Example:
            >>> pattern = HistoricalPattern(name="Test", preconditions={'ECONOMIC_wealth_inequality': (0.7, 1.0)})
            >>> pattern.get_metric_weight('ECONOMIC_wealth_inequality')
            0.3
        """
        weights = {
            'ECONOMIC_wealth_inequality': 0.3,
            'POLITICAL_institutional_trust': 0.25,
            'SOCIAL_civic_engagement': 0.2,
            'ENVIRONMENTAL_climate_stress': 0.15,
            'TECHNOLOGICAL_innovation_rate': 0.1
        }
        return weights.get(metric_key, 1.0 / len(self.preconditions))

class PatternManager:
    def __init__(self):
        self.patterns = [
            HistoricalPattern(
                name="Economic Collapse",
                preconditions={
                    'ECONOMIC_wealth_inequality': (0.7, 1.0),
                    'ECONOMIC_economic_stability': (0.0, 0.3)
                },
                outcome="Potential economic crisis due to high inequality",
                severity=0.9,
                timeframe="short_term"
            ),
            HistoricalPattern(
                name="Political Instability",
                preconditions={
                    'POLITICAL_institutional_trust': (0.0, 0.4),
                    'SOCIAL_civic_engagement': (0.0, 0.5)
                },
                outcome="Risk of governance challenges or unrest",
                severity=0.8,
                timeframe="medium_term"
            ),
            HistoricalPattern(
                name="Technological Singularity",
                preconditions={
                    'TECHNOLOGICAL_innovation_rate': (0.8, 1.0),
                    'SOCIAL_education_level': (0.7, 1.0)
                },
                outcome="Rapid technological transformation",
                severity=0.7,
                timeframe="long_term"
            )
        ]

    def get_active_patterns(self) -> List[HistoricalPattern]:
        """Return all active historical patterns."""
        return self.patterns

    def get_patterns_by_timeframe(self, timeframe: str) -> List[HistoricalPattern]:
        """Return patterns matching the specified timeframe."""
        return [p for p in self.patterns if p.timeframe == timeframe]

class CivilizationMetrics:
    def __init__(self):
        self.historical_data = []
        self.current_state = {cat.value: {} for cat in MetricCategory}
        self.metric_ranges = {cat.value: {} for cat in MetricCategory}

    def take_snapshot(self, date: datetime):
        """Store a snapshot of current metrics and trends.

        Args:
            date (datetime): Timestamp for the snapshot

        Raises:
            ValueError: If no metrics are available
        """
        if not any(self.current_state.values()):
            raise ValueError("No metrics available for snapshot")
        self.historical_data.append({
            'date': date,
            'metrics': {cat: dict(metrics) for cat, metrics in self.current_state.items()},
            'trends': self._calculate_trends()
        })

    def _calculate_trends(self) -> Dict:
        """Calculate trends based on differences between historical snapshots."""
        trends = {cat: {} for cat in self.current_state}
        if len(self.historical_data) < 2:
            return trends

        prev_data = self.historical_data[-2]['metrics']
        curr_data = self.historical_data[-1]['metrics'] if self.historical_data else self.current_state

        for category in curr_data:
            for metric, curr_value in curr_data[category].items():
                prev_value = prev_data.get(category, {}).get(metric, curr_value)
                trends[category][metric] = curr_value - prev_value

        return trends

    def clone(self):
        """Efficiently clone metrics without deep copying."""
        new_obj = CivilizationMetrics()
        new_obj.current_state = {cat: dict(metrics) for cat, metrics in self.current_state.items()}
        new_obj.metric_ranges = {cat: dict(ranges) for cat, ranges in self.metric_ranges.items()}
        new_obj.historical_data = [dict(d) for d in self.historical_data]
        return new_obj

    def update_metric(self, category: MetricCategory, metric: str, value: float, min_val: float = 0.0, max_val: float = 1.0):
        """Update a metric with normalization and validation.

        Args:
            category (MetricCategory): Metric category
            metric (str): Metric name
            value (float): Raw metric value
            min_val (float): Minimum value for normalization
            max_val (float): Maximum value for normalization

        Raises:
            TypeError: If value is not numeric
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Metric value for {metric} must be numeric, got {type(value)}")
        normalized_value = MetricNormalizer.normalize(value, min_val, max_val)
        self.current_state[category.value][metric] = normalized_value
        self.metric_ranges[category.value][metric] = (min_val, max_val)

    def get_metric(self, category: MetricCategory, metric: str) -> float:
        """Retrieve a metric value."""
        return self.current_state.get(category.value, {}).get(metric, 0.0)

class MetricNormalizer:
    @staticmethod
    def normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to [0,1] range."""
        if max_val == min_val:
            logging.warning(f"Min and max values are equal for normalization: {min_val}")
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    @staticmethod
    def denormalize(normalized: float, min_val: float, max_val: float) -> float:
        """Convert a normalized value back to its original range."""
        return min_val + normalized * (max_val - min_val)

class UncertaintyQuantifier:
    def analyze_uncertainty(self, metrics: Dict, patterns: List[Dict], risk_score: float) -> Dict:
        """Quantify uncertainty using Monte Carlo simulation.

        Args:
            metrics (Dict): Normalized metrics
            patterns (List[Dict]): Matched patterns
            risk_score (float): Base risk score

        Returns:
            Dict: Uncertainty analysis with confidence interval and sensitivity
        """
        n_simulations = 1000
        simulated_scores = []
        for _ in range(n_simulations):
            noise = np.random.normal(0, 0.05, len(metrics['values']))
            noisy_metrics = {
                k: max(0.0, min(1.0, v + n)) for k, v, n in zip(metrics['values'].keys(), metrics['values'].values(), noise)
            }
            simulated_score = self._simulate_risk(noisy_metrics, patterns)
            simulated_scores.append(simulated_score)

        mean_score = np.mean(simulated_scores)
        std_score = np.std(simulated_scores)
        return {
            'confidence_interval': (max(0.0, mean_score - 1.96 * std_score), min(1.0, mean_score + 1.96 * std_score)),
            'sensitivity_analysis': self._compute_sensitivity(metrics, patterns),
            'monte_carlo_std': std_score
        }

    def _simulate_risk(self, metrics: Dict, patterns: List[Dict]) -> float:
        """Simplified risk simulation for uncertainty analysis."""
        base_risk = np.mean([v for k, v in metrics.items() if 'inequality' in k or 'stress' in k], initial=0.5)
        pattern_risk = max((p['severity'] * p['match_score'] for p in patterns), default=0.0)
        return min(1.0, 0.6 * base_risk + 0.4 * pattern_risk)

    def _compute_sensitivity(self, metrics: Dict, patterns: List[Dict]) -> Dict:
        """Compute sensitivity of risk to each metric."""
        sensitivity = {}
        baseline_risk = self._simulate_risk(metrics, patterns)
        for key in metrics:
            temp_metrics = metrics.copy()
            temp_metrics[key] = min(1.0, temp_metrics[key] + 0.1)
            new_risk = self._simulate_risk(temp_metrics, patterns)
            sensitivity[key] = (new_risk - baseline_risk) / 0.1
        return sensitivity

# ==================== CONFIGURATION & CONSTANTS ====================

class EngineConfig:
    RISK_WEIGHTS = {
        'base_metrics': 0.5,
        'pattern_matches': 0.3,
        'trend_momentum': 0.2
    }
    PROJECTION_WEIGHTS = {'current': 0.6, 'projected': 0.4}
    TREND_DECAY_RATE = 0.7
    MAX_PROJECTION_YEARS = 10
    RISK_THRESHOLDS = {
        'CRITICAL': 0.95,
        'HIGH': 0.8,
        'MEDIUM': 0.6,
        'LOW': 0.4,
        'STABLE': 0.0
    }
    PATTERN_MATCH_THRESHOLD = 0.6
    PARTIAL_MATCH_DECAY = 5.0
    INTERVENTION_DIMINISHING_RETURNS = 0.3
    MIN_INTERVENTION_EFFECT = 0.05
    SIDE_EFFECT_RULES = {
        ('ECONOMIC', 'wealth_inequality'): [
            ('POLITICAL', 'social_unrest', 0.3),
            ('SOCIAL', 'civic_engagement', -0.2)
        ],
        ('POLITICAL', 'institutional_trust'): [
            ('SOCIAL', 'civic_engagement', 0.4),
            ('ECONOMIC', 'economic_stability', 0.3)
        ]
    }
    CACHE_SIZE_LIMITS = {
        'normalization': 1000,
        'pattern_matching': 500
    }
    LOG_LEVEL = "INFO"

    @classmethod
    def validate(cls):
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if abs(sum(cls.RISK_WEIGHTS.values()) - 1.0) > 1e-6:
            raise ValueError("Risk weights must sum to 1.0")
        if cls.MAX_PROJECTION_YEARS <= 0:
            raise ValueError("Max projection years must be positive")
        if not (0 < cls.PATTERN_MATCH_THRESHOLD <= 1):
            raise ValueError("Pattern match threshold must be in (0,1]")
        if not all(v > 0 for v in cls.CACHE_SIZE_LIMITS.values()):
            raise ValueError("Cache size limits must be positive")
        logging.info("EngineConfig validated successfully")

# ==================== DATA STRUCTURES ====================

@dataclass
class AnalysisResult:
    date: datetime
    risk_score: float
    risk_level: str
    pattern_matches: List[Dict]
    recommendations: List[Dict]
    metric_snapshot: Dict[str, float]
    metric_trends: Dict[str, float]
    uncertainty_analysis: Dict = field(default_factory=dict)
    projection_confidence: float = 0.0

@dataclass
class TimelinePrediction:
    timeframe: str
    year: int
    risk_score: float
    risk_level: str
    predictions: List[Dict]
    key_events: List[Dict]
    confidence: float

@dataclass
class InterventionResult:
    baseline_risk: float
    projected_risk: float
    risk_change: float
    intervention_effectiveness: float
    projected_patterns: List[Dict]
    time_horizon: int
    cost_benefit_ratio: float = 0.0
    success_probability: float = 0.0

# ==================== CORE ENGINE ====================

class PsychohistoryEngine:
    """
    Production-ready civilizational analysis engine with accurate prediction capabilities.

    Attributes:
        pattern_manager (PatternManager): Manages historical patterns
        uncertainty_quantifier (UncertaintyQuantifier): Quantifies prediction uncertainty
        civilizations (Dict): Registered civilizations with metrics and history
        temporal_resolution (timedelta): Time interval for snapshots
        config (EngineConfig): Configuration parameters
        prediction_models (Dict): Available prediction models
        _normalize_cache (OrderedDict): Cache for normalized metrics
        _pattern_cache (OrderedDict): Cache for pattern matching results
    """
    def __init__(self, config_class=EngineConfig):
        config_class.validate()
        setup_logging(config_class.LOG_LEVEL)
        self.pattern_manager = PatternManager()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.civilizations = {}
        self.temporal_resolution = timedelta(days=90)
        self.config = config_class
        self.prediction_models = {
            'linear': PredictionModel.linear_projection,
            'logistic': PredictionModel.logistic_growth,
            'momentum': PredictionModel.momentum_weighted
        }
        self._normalize_cache = OrderedDict()
        self._pattern_cache = OrderedDict()

    def _prune_cache(self, cache: OrderedDict, limit: int):
        """Prune cache to maintain size limit."""
        while len(cache) > limit:
            cache.popitem(last=False)

    def add_civilization(self, name: str, metrics: CivilizationMetrics) -> None:
        """Register a civilization with validation.

        Args:
            name (str): Unique name of the civilization
            metrics (CivilizationMetrics): Metrics object for the civilization

        Raises:
            ValueError: If name is invalid
            TypeError: If metrics is not a CivilizationMetrics instance
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Civilization name must be a non-empty string")
        if not isinstance(metrics, CivilizationMetrics):
            raise TypeError("metrics must be a CivilizationMetrics instance")
        self.civilizations[name] = {
            'metrics': metrics,
            'analyses': [],
            'risk_history': [],
            'intervention_history': []
        }
        logging.info(f"Added civilization: {name}")

    def analyze_civilization(self, civ_name: str, analysis_date: datetime = None) -> AnalysisResult:
        """Perform comprehensive analysis with uncertainty quantification.

        Args:
            civ_name (str): Name of the civilization
            analysis_date (datetime, optional): Date for analysis. Defaults to current time.

        Returns:
            AnalysisResult: Analysis results including risk score, patterns, and recommendations

        Example:
            >>> engine = PsychohistoryEngine()
            >>> metrics = CivilizationMetrics()
            >>> metrics.update_metric(MetricCategory.ECONOMIC, "wealth_inequality", 70, 0, 100)
            >>> engine.add_civilization("Earth2025", metrics)
            >>> result = engine.analyze_civilization("Earth2025")
        """
        self._validate_civilization(civ_name)
        if analysis_date is None:
            analysis_date = datetime.now()
        civ = self.civilizations[civ_name]
        result = self._run_analysis_logic(civ['metrics'], analysis_date)
        civ['analyses'].append(result)
        civ['risk_history'].append((analysis_date, result.risk_score))
        return result

    def analyze_multiple_civilizations(self, civ_names: List[str], analysis_date: datetime = None) -> Dict[str, AnalysisResult]:
        """Batch analyze multiple civilizations for performance.

        Args:
            civ_names (List[str]): List of civilization names
            analysis_date (datetime, optional): Date for analysis. Defaults to current time.

        Returns:
            Dict[str, AnalysisResult]: Analysis results for each civilization
        """
        results = {}
        for name in civ_names:
            try:
                results[name] = self.analyze_civilization(name, analysis_date)
            except ValueError as e:
                logging.error(f"Failed to analyze {name}: {e}")
        return results

    def _run_analysis_logic(self, metrics: CivilizationMetrics, analysis_date: datetime) -> AnalysisResult:
        """Core analysis logic decoupled from storage."""
        metrics = self._prepare_metrics(metrics, analysis_date)
        normalized = self._normalize_metrics(metrics.historical_data[-1]['metrics'], metrics.historical_data[-1].get('trends', {}))
        pattern_matches = self._match_patterns(normalized)
        risk_score = self._calculate_composite_risk(normalized, pattern_matches)
        risk_level = self._determine_risk_level(risk_score)
        recommendations = self._generate_recommendations(normalized, pattern_matches)
        uncertainty = self.uncertainty_quantifier.analyze_uncertainty(normalized, pattern_matches, risk_score)
        return AnalysisResult(
            date=analysis_date,
            risk_score=risk_score,
            risk_level=risk_level,
            pattern_matches=pattern_matches,
            recommendations=recommendations,
            metric_snapshot=normalized['values'],
            metric_trends=normalized['trends'],
            uncertainty_analysis=uncertainty,
            projection_confidence=uncertainty.get('confidence', 0.8)
        )

    def _prepare_metrics(self, metrics: CivilizationMetrics, analysis_date: datetime) -> CivilizationMetrics:
        """Prepare metrics for analysis by taking a snapshot if needed."""
        if (not metrics.historical_data or 
            (analysis_date - metrics.historical_data[-1]['date']) >= self.temporal_resolution):
            metrics.take_snapshot(analysis_date)
        if not metrics.historical_data:
            raise ValueError("No historical data available for analysis")
        return metrics

    def predict_timeline(self, civ_name: str, years: int = 5, include_interventions: List[Dict] = None) -> List[TimelinePrediction]:
        """Generate multi-year predictions with trend propagation and intervention modeling.

        Args:
            civ_name (str): Civilization name
            years (int): Number of years to project
            include_interventions (List[Dict], optional): List of interventions to apply

        Returns:
            List[TimelinePrediction]: Predicted timeline with risk scores and events
        """
        self._validate_civilization(civ_name)
        if years > self.config.MAX_PROJECTION_YEARS:
            raise ValueError(f"Cannot project more than {self.config.MAX_PROJECTION_YEARS} years")
        civ = self.civilizations[civ_name]
        base_metrics = civ['metrics'].clone()
        timeline = []
        current_metrics = base_metrics.clone()

        for year in range(1, years + 1):
            if include_interventions:
                current_metrics = self._apply_interventions(current_metrics, include_interventions, year)
            projected_metrics = self._project_metrics_forward(current_metrics, years=1)
            projection_date = datetime.now() + timedelta(days=365 * year)
            projection_analysis = self._run_analysis_logic(projected_metrics, projection_date)
            yearly_predictions = self._generate_year_predictions(projection_analysis, year)
            key_events = self._predict_key_events(projection_analysis, year)
            timeline.append(TimelinePrediction(
                timeframe=f"Year +{year}",
                year=year,
                risk_score=projection_analysis.risk_score,
                risk_level=projection_analysis.risk_level,
                predictions=yearly_predictions,
                key_events=key_events,
                confidence=projection_analysis.projection_confidence
            ))
            current_metrics = projected_metrics

        return timeline

    def _project_metrics_forward(self, metrics: CivilizationMetrics, years: int) -> CivilizationMetrics:
        """Project all metrics forward using vectorized operations.

        Args:
            metrics (CivilizationMetrics): Input metrics
            years (int): Projection horizon

        Returns:
            CivilizationMetrics: Projected metrics
        """
        projected_metrics = metrics.clone()
        current_state = projected_metrics.historical_data[-1]['metrics'] if projected_metrics.historical_data else projected_metrics.current_state
        current_trends = projected_metrics.historical_data[-1].get('trends', {}) if projected_metrics.historical_data else {}

        # Vectorize metric projections
        keys, values, trends = [], [], []
        for category, metric_dict in current_state.items():
            for metric_name, value in metric_dict.items():
                keys.append((category, metric_name))
                values.append(value)
                trends.append(current_trends.get(category, {}).get(metric_name, 0.0))

        values = np.array(values)
        trends = np.array(trends)
        projected_values = np.zeros_like(values)

        # Apply appropriate prediction model per metric
        for i, (category, metric_name) in enumerate(keys):
            if 'growth' in metric_name.lower() or 'adoption' in metric_name.lower():
                projected_values[i] = self.prediction_models['logistic'](
                    values[i], capacity=1.0, growth_rate=trends[i], years=years
                )
            elif abs(trends[i]) > 0.2:
                historical_trends = self._get_historical_trends(metrics, f"{category}_{metric_name}")
                projected_values[i] = self.prediction_models['momentum'](
                    values[i], historical_trends, years
                )
            else:
                projected_values[i] = self.prediction_models['linear'](
                    values[i], trends[i], years, self.config.TREND_DECAY_RATE
                )

        # Update projected metrics
        for (category, metric_name), value in zip(keys, projected_values):
            projected_metrics.update_metric(MetricCategory[category.upper()], metric_name, value)

        projected_date = datetime.now() + timedelta(days=365 * years)
        projected_metrics.take_snapshot(projected_date)
        return projected_metrics

    def _get_historical_trends(self, metrics: CivilizationMetrics, metric_key: str) -> List[float]:
        """Extract historical trends for a specific metric."""
        trends = []
        if len(metrics.historical_data) < 2:
            return trends
        for i in range(1, len(metrics.historical_data)):
            current = self._get_metric_value(metrics.historical_data[i]['metrics'], metric_key)
            previous = self._get_metric_value(metrics.historical_data[i-1]['metrics'], metric_key)
            if current is not None and previous is not None:
                trends.append(current - previous)
        return trends

    def _get_metric_value(self, state: Dict, metric_key: str) -> Optional[float]:
        """Extract metric value from nested state structure."""
        if '_' in metric_key:
            category, metric_name = metric_key.split('_', 1)
            return state.get(category, {}).get(metric_name)
        return None

    def simulate_intervention(self, civ_name: str, intervention: Dict[str, float], years: int = 5) -> InterventionResult:
        """Simulate intervention with diminishing returns and side effects.

        Args:
            civ_name (str): Civilization name
            intervention (Dict[str, float]): Metric changes (e.g., {'ECONOMIC_wealth_inequality': -0.2})
            years (int): Projection horizon

        Returns:
            InterventionResult: Results including risk change and effectiveness
        """
        self._validate_civilization(civ_name)
        baseline_analysis = self.analyze_civilization(civ_name)
        base_metrics = self.civilizations[civ_name]['metrics'].clone()
        intervened_metrics = self._apply_intervention_effects(base_metrics, intervention)
        projected_metrics = self._project_metrics_forward(intervened_metrics, years)
        projection_date = datetime.now() + timedelta(days=365 * years)
        projected_analysis = self._run_analysis_logic(projected_metrics, projection_date)
        risk_change = projected_analysis.risk_score - baseline_analysis.risk_score
        effectiveness = -risk_change
        success_prob = self._calculate_success_probability(intervention, baseline_analysis.risk_score)
        return InterventionResult(
            baseline_risk=baseline_analysis.risk_score,
            projected_risk=projected_analysis.risk_score,
            risk_change=risk_change,
            intervention_effectiveness=effectiveness,
            projected_patterns=projected_analysis.pattern_matches,
            time_horizon=years,
            success_probability=success_prob,
            cost_benefit_ratio=self._calculate_cost_benefit(intervention, effectiveness)
        )

    def _apply_intervention_effects(self, metrics: CivilizationMetrics, intervention: Dict[str, float]) -> CivilizationMetrics:
        """Apply intervention effects with diminishing returns and side effects."""
        intervened_metrics = metrics.clone()
        for intervention_key, effect in intervention.items():
            if '_' not in intervention_key:
                logging.warning(f"Invalid intervention key: {intervention_key}")
                continue
            category_str, metric_name = intervention_key.split('_', 1)
            try:
                category = MetricCategory[category_str.upper()]
                current_value = intervened_metrics.get_metric(category, metric_name)
                if current_value is None:
                    logging.warning(f"Metric {intervention_key} not found")
                    continue
                # Apply diminishing returns to intervention effect
                diminishing_factor = 1 - (abs(effect) * self.config.INTERVENTION_DIMINISHING_RETURNS)
                adjusted_effect = effect * diminishing_factor
                if abs(adjusted_effect) < self.config.MIN_INTERVENTION_EFFECT and effect != 0:
                    adjusted_effect = np.sign(effect) * self.config.MIN_INTERVENTION_EFFECT
                new_value = max(0.0, min(1.0, current_value + adjusted_effect))
                intervened_metrics.update_metric(category, metric_name, new_value)
                self._apply_side_effects(intervened_metrics, category, metric_name, effect)
            except (KeyError, AttributeError) as e:
                logging.warning(f"Error applying intervention {intervention_key}: {e}")
                continue
        return intervened_metrics

    def _apply_side_effects(self, metrics: CivilizationMetrics, category: MetricCategory, metric_name: str, primary_effect: float):
        """Apply side effects to related metrics based on configuration rules."""
        key = (category.name, metric_name)
        rules = self.config.SIDE_EFFECT_RULES.get(key, [])
        for eff_category, eff_metric, multiplier in rules:
            try:
                current_val = metrics.get_metric(MetricCategory[eff_category], eff_metric)
                side_effect = primary_effect * multiplier * 0.5
                new_val = max(0.0, min(1.0, current_val + side_effect))
                metrics.update_metric(MetricCategory[eff_category], eff_metric, new_val)
            except KeyError:
                logging.warning(f"Side effect metric {eff_category}_{eff_metric} not found")

    def _apply_interventions(self, metrics: CivilizationMetrics, interventions: List[Dict], years_elapsed: int) -> CivilizationMetrics:
        """Apply multiple interventions with temporal effects."""
        intervened_metrics = metrics.clone()
        for intervention in interventions:
            start_year = intervention.get('start_year', 0)
            if years_elapsed >= start_year:
                decay_factor = intervention.get('decay_rate', 1.0) ** years_elapsed
                scaled_intervention = {
                    k: v * decay_factor for k, v in intervention.items() if k not in ['start_year', 'decay_rate']
                }
                intervened_metrics = self._apply_intervention_effects(intervened_metrics, scaled_intervention)
        return intervened_metrics

    def _calculate_success_probability(self, intervention: Dict, current_risk: float) -> float:
        """Calculate intervention success probability based on magnitude and context."""
        total_magnitude = sum(abs(v) for v in intervention.values())
        risk_factor = 1.0 - (current_risk * 0.5)
        base_prob = min(0.9, total_magnitude * 2.0)
        return max(0.1, base_prob * risk_factor)

    def _calculate_cost_benefit(self, intervention: Dict, effectiveness: float) -> float:
        """Calculate cost-benefit ratio of intervention."""
        total_cost = sum(abs(v) for v in intervention.values())
        return float('inf') if total_cost == 0 else effectiveness / total_cost

    def _generate_year_predictions(self, analysis: AnalysisResult, year: int) -> List[Dict]:
        """Generate specific predictions for a given year."""
        predictions = []
        confidence_decay = max(0.1, 1.0 - (year * 0.08))
        for match in analysis.pattern_matches[:3]:
            predictions.append({
                'type': 'pattern_based',
                'outcome': match['predicted_outcome'],
                'confidence': match['match_score'] * confidence_decay,
                'source_pattern': match['pattern_name'],
                'timeframe': f"{year} years"
            })
        if analysis.risk_score > 0.8:
            predictions.append({
                'type': 'risk_based',
                'outcome': 'High probability of systemic instability or crisis',
                'confidence': analysis.risk_score * 0.8,
                'trigger_metrics': self._identify_risk_triggers(analysis),
                'timeframe': f"{year} years"
            })
        elif analysis.risk_score < 0.3:
            predictions.append({
                'type': 'risk_based',
                'outcome': 'Continued stability with gradual improvement likely',
                'confidence': (1.0 - analysis.risk_score) * 0.7,
                'supporting_metrics': self._identify_stability_factors(analysis),
                'timeframe': f"{year} years"
            })
        return predictions

    def _predict_key_events(self, analysis: AnalysisResult, year: int) -> List[Dict]:
        """Predict key events based on patterns and metrics."""
        events = []
        for pattern in analysis.pattern_matches:
            if pattern['match_score'] > 0.7:
                events.append({
                    'type': pattern['pattern_name'].lower().replace(' ', '_'),
                    'description': pattern['predicted_outcome'],
                    'confidence': pattern['match_score'] * 0.9,
                    'year': year
                })
        for metric, value in analysis.metric_snapshot.items():
            if ('inequality' in metric or 'stress' in metric) and value > 0.7:
                events.append({
                    'type': f"{metric.split('_')[0].lower()}_crisis",
                    'description': f"Potential {metric.split('_')[1]} crisis",
                    'confidence': value * 0.8,
                    'year': year
                })
            elif ('trust' in metric or 'engagement' in metric) and value < 0.3:
                events.append({
                    'type': f"{metric.split('_')[0].lower()}_instability",
                    'description': f"Risk of {metric.split('_')[1]} decline",
                    'confidence': (1.0 - value) * 0.7,
                    'year': year
                })
        return events

    def _identify_risk_triggers(self, analysis: AnalysisResult) -> List[str]:
        """Identify metrics contributing most to high risk."""
        critical_metrics = []
        for metric, value in analysis.metric_snapshot.items():
            if ('inequality' in metric or 'stress' in metric) and value > 0.7:
                critical_metrics.append(metric)
            elif ('trust' in metric or 'engagement' in metric) and value < 0.3:
                critical_metrics.append(metric)
        return critical_metrics

    def _identify_stability_factors(self, analysis: AnalysisResult) -> List[str]:
        """Identify metrics contributing to stability."""
        stability_metrics = []
        for metric, value in analysis.metric_snapshot.items():
            if ('stability' in metric or 'growth' in metric) and value > 0.6:
                stability_metrics.append(metric)
            elif ('trust' in metric or 'engagement' in metric) and value > 0.6:
                stability_metrics.append(metric)
        return stability_metrics

    def _validate_civilization(self, civ_name: str) -> None:
        """Validate civilization exists."""
        if civ_name not in self.civilizations:
            logging.error(f"Unknown civilization: {civ_name}")
            raise ValueError(f"Unknown civilization: {civ_name}")

    def _normalize_metrics(self, values: Dict, trends: Dict) -> Dict:
        """Normalize metrics with caching for performance.

        Args:
            values (Dict): Raw metric values
            trends (Dict): Metric trends

        Returns:
            Dict: Normalized metrics with values, trends, and projected values
        """
        cache_key = str(hash(str(values) + str(trends)))
        if cache_key in self._normalize_cache:
            return self._normalize_cache[cache_key]
        normalized = {'values': {}, 'trends': {}, 'projected': {}}
        c_w, p_w = self.config.PROJECTION_WEIGHTS['current'], self.config.PROJECTION_WEIGHTS['projected']
        for category, metrics_dict in values.items():
            for metric, value in metrics_dict.items():
                key = f"{category}_{metric}"
                try:
                    value = max(0.0, min(1.0, float(value)))
                    trend = float(trends.get(category, {}).get(metric, 0.0))
                    normalized['values'][key] = value
                    normalized['trends'][key] = trend
                    # Weighted projection of current and trend-based values
                    projected = (value * c_w) + ((value + trend) * p_w)
                    normalized['projected'][key] = max(0.0, min(1.0, projected))
                except (TypeError, ValueError) as e:
                    logging.warning(f"Invalid metric {key}: {e}")
                    continue
        self._normalize_cache[cache_key] = normalized
        self._prune_cache(self._normalize_cache, self.config.CACHE_SIZE_LIMITS['normalization'])
        return normalized

    def _match_patterns(self, metrics: Dict) -> List[Dict]:
        """Match current state against historical patterns with caching."""
        matches = []
        for pattern in self.pattern_manager.get_active_patterns():
            match_score = self._calculate_pattern_match(metrics, pattern)
            if match_score >= self.config.PATTERN_MATCH_THRESHOLD:
                matches.append({
                    'pattern_name': pattern.name,
                    'match_score': match_score,
                    'predicted_outcome': pattern.outcome,
                    'timeframe': pattern.timeframe,
                    'severity': pattern.severity,
                    'confidence': pattern.confidence_threshold
                })
        matches.sort(key=lambda x: (x['severity'], x['match_score']), reverse=True)
        return matches

    def _calculate_pattern_match(self, metrics: Dict, pattern: HistoricalPattern) -> float:
        """Calculate pattern match score with caching.

        Args:
            metrics (Dict): Normalized metrics
            pattern (HistoricalPattern): Pattern to match against

        Returns:
            float: Match score in [0,1]
        """
        cache_key = f"pattern_{pattern.name}_{hash(str(metrics))}"
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

        total_weight = 0.0
        matched_weight = 0.0
        c_w = self.config.PROJECTION_WEIGHTS['current']
        p_w = self.config.PROJECTION_WEIGHTS['projected']

        # Iterate through pattern preconditions
        for metric_key, (min_val, max_val) in pattern.preconditions.items():
            if metric_key not in metrics['values']:
                logging.warning(f"Metric {metric_key} not found in metrics")
                continue
            weight = pattern.get_metric_weight(metric_key)
            total_weight += weight
            current_value = metrics['values'][metric_key]
            projected_value = metrics['projected'][metric_key]
            # Combine current and projected values for matching
            effective_value = current_value * c_w + projected_value * p_w
            if min_val <= effective_value <= max_val:
                matched_weight += weight
            else:
                # Partial match with exponential decay for distance
                distance = min_val - effective_value if effective_value < min_val else effective_value - max_val
                partial_match = np.exp(-distance * self.config.PARTIAL_MATCH_DECAY)
                matched_weight += weight * partial_match

        result = matched_weight / total_weight if total_weight > 0 else 0.0
        self._pattern_cache[cache_key] = result
        self._prune_cache(self._pattern_cache, self.config.CACHE_SIZE_LIMITS['pattern_matching'])
        return result

    def _calculate_composite_risk(self, metrics: Dict, pattern_matches: List[Dict]) -> float:
        """Calculate comprehensive risk score.

        Combines base metric risk, pattern match risk, and trend momentum risk.

        Args:
            metrics (Dict): Normalized metrics
            pattern_matches (List[Dict]): Matched patterns

        Returns:
            float: Composite risk score in [0,1]
        """
        critical_metrics = [
            'ECONOMIC_wealth_inequality', 'POLITICAL_institutional_trust',
            'SOCIAL_civic_engagement', 'ENVIRONMENTAL_climate_stress'
        ]
        metric_risks = []
        for metric in critical_metrics:
            if metric in metrics['values']:
                value = metrics['values'][metric]
                # Invert trust/engagement metrics for risk (low value = high risk)
                risk = 1.0 - value if 'trust' in metric or 'engagement' in metric else value
                metric_risks.append(risk)
        base_risk = np.mean(metric_risks) if metric_risks else 0.5
        pattern_risk = max([match['severity'] * match['match_score'] * match['confidence'] for match in pattern_matches], default=0.0)
        negative_trends = [t for t in metrics['trends'].values() if t < -0.1]
        trend_risk = abs(np.mean(negative_trends)) if negative_trends else 0.0
        weights = self.config.RISK_WEIGHTS
        composite = (base_risk * weights['base_metrics'] + 
                     pattern_risk * weights['pattern_matches'] + 
                     trend_risk * weights['trend_momentum'])
        return min(1.0, composite)

    def _determine_risk_level(self, score: float) -> str:
        """Convert risk score to categorical level."""
        thresholds = self.config.RISK_THRESHOLDS
        if score >= thresholds['CRITICAL']:
            return "CRITICAL"
        elif score >= thresholds['HIGH']:
            return "HIGH"
        elif score >= thresholds['MEDIUM']:
            return "MEDIUM"
        elif score >= thresholds['LOW']:
            return "LOW"
        return "STABLE"

    def _generate_recommendations(self, metrics: Dict, pattern_matches: List[Dict]) -> List[Dict]:
        """Generate actionable recommendations based on patterns and metrics."""
        recommendations = []
        for pattern in pattern_matches:
            if pattern['severity'] > 0.7:
                recommendations.append({
                    'pattern': pattern['pattern_name'],
                    'action': self._get_recommendation_for_pattern(pattern['pattern_name']),
                    'priority': pattern['severity'],
                    'confidence': pattern['confidence']
                })
        for metric, value in metrics['values'].items():
            if ('inequality' in metric or 'stress' in metric) and value > 0.7:
                recommendations.append({
                    'metric': metric,
                    'action': f"Implement policies to reduce {metric.split('_')[1]}",
                    'priority': value,
                    'confidence': 0.8
                })
            elif ('trust' in metric or 'engagement' in metric) and value < 0.3:
                recommendations.append({
                    'metric': metric,
                    'action': f"Enhance {metric.split('_')[1]} through transparency and engagement",
                    'priority': 1.0 - value,
                    'confidence': 0.8
                })
        return recommendations

    def _get_recommendation_for_pattern(self, pattern_name: str) -> str:
        """Map pattern names to specific recommendations."""
        recommendations = {
            "Economic Collapse": "Implement progressive taxation and social safety nets",
            "Political Instability": "Strengthen democratic institutions and public communication",
            "Technological Singularity": "Invest in ethical AI governance and education"
        }
        return recommendations.get(pattern_name, "Review and address underlying metrics")

    def plot_risk_breakdown(self, analysis: AnalysisResult):
        """Visualize risk components as a pie chart.

        Args:
            analysis (AnalysisResult): Analysis result to visualize
        """
        components = {
            'Base Metrics': analysis.risk_score * self.config.RISK_WEIGHTS['base_metrics'],
            'Pattern Matches': analysis.risk_score * self.config.RISK_WEIGHTS['pattern_matches'],
            'Trend Momentum': analysis.risk_score * self.config.RISK_WEIGHTS['trend_momentum']
        }
        labels = list(components.keys())
        sizes = list(components.values())
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title(f"Risk Breakdown - {analysis.risk_level} (Score: {analysis.risk_score:.3f})")
        plt.axis('equal')
        plt.show()

# ==================== USAGE EXAMPLE AND TESTING ====================

def main():
    """Demonstrate the Psychohistory Engine with realistic data and visualizations."""
    print("=== Psychohistory Engine - Production Ready ===")
    engine = PsychohistoryEngine()
    metrics = CivilizationMetrics()

    # Initialize realistic metrics (e.g., wealth inequality as Gini coefficient * 100)
    metrics.update_metric(MetricCategory.ECONOMIC, "wealth_inequality", 70, 0, 100)
    metrics.update_metric(MetricCategory.ECONOMIC, "economic_stability", 30, 0, 100)
    metrics.update_metric(MetricCategory.POLITICAL, "institutional_trust", 40, 0, 100)
    metrics.update_metric(MetricCategory.SOCIAL, "civic_engagement", 50, 0, 100)
    metrics.update_metric(MetricCategory.ENVIRONMENTAL, "climate_stress", 60, 0, 100)
    metrics.update_metric(MetricCategory.TECHNOLOGICAL, "innovation_rate", 80, 0, 100)
    metrics.update_metric(MetricCategory.SOCIAL, "education_level", 75, 0, 100)
    metrics.take_snapshot(datetime.now())
    engine.add_civilization("Earth2025", metrics)

    # Run analysis
    logging.info("Running Analysis...")
    analysis = engine.analyze_civilization("Earth2025")
    print(f"Risk Score: {analysis.risk_score:.3f} ({analysis.risk_level})")
    print(f"Pattern Matches: {len(analysis.pattern_matches)}")
    print("Recommendations:", [r['action'] for r in analysis.recommendations])

    # Visualize risk breakdown
    logging.info("Generating Risk Breakdown Plot...")
    engine.plot_risk_breakdown(analysis)

    # Generate timeline predictions
    logging.info("Generating Timeline Predictions...")
    timeline = engine.predict_timeline("Earth2025", years=3)
    years = [p.year for p in timeline]
    risks = [p.risk_score for p in timeline]
    plt.figure(figsize=(8, 6))
    plt.plot(years, risks, marker='o', label='Risk Score')
    plt.title("Risk Score Over Time")
    plt.xlabel("Years")
    plt.ylabel("Risk Score")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Simulate intervention
    logging.info("Simulating Intervention...")
    intervention = {
        "ECONOMIC_wealth_inequality": -0.2,
        "POLITICAL_institutional_trust": 0.15
    }
    result = engine.simulate_intervention("Earth2025", intervention, years=3)
    print(f"Baseline Risk: {result.baseline_risk:.3f}")
    print(f"Projected Risk: {result.projected_risk:.3f}")
    print(f"Risk Change: {result.risk_change:+.3f}")
    print(f"Success Probability: {result.success_probability:.1%}")

class TestPsychohistoryEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PsychohistoryEngine()
        self.metrics = CivilizationMetrics()
        self.metrics.update_metric(MetricCategory.ECONOMIC, "wealth_inequality", 70, 0, 100)
        self.metrics.update_metric(MetricCategory.POLITICAL, "institutional_trust", 40, 0, 100)
        self.metrics.take_snapshot(datetime.now())
        self.engine.add_civilization("TestCiv", self.metrics)

    def test_analyze_civilization(self):
        analysis = self.engine.analyze_civilization("TestCiv")
        self.assertTrue(0.0 <= analysis.risk_score <= 1.0)
        self.assertIn(analysis.risk_level, ["STABLE", "LOW", "MEDIUM", "HIGH", "CRITICAL"])

    def test_predict_timeline(self):
        timeline = self.engine.predict_timeline("TestCiv", years=2)
        self.assertEqual(len(timeline), 2)
        self.assertTrue(all(0.0 <= p.risk_score <= 1.0 for p in timeline))

    def test_invalid_civilization(self):
        with self.assertRaises(ValueError):
            self.engine.analyze_civilization("InvalidCiv")

    def test_intervention_simulation(self):
        intervention = {"ECONOMIC_wealth_inequality": -0.1}
        result = self.engine.simulate_intervention("TestCiv", intervention, years=2)
        self.assertTrue(0.0 <= result.projected_risk <= 1.0)
        self.assertTrue(0.0 <= result.success_probability <= 1.0)

    def test_batch_analysis(self):
        self.engine.add_civilization("TestCiv2", self.metrics.clone())
        results = self.engine.analyze_multiple_civilizations(["TestCiv", "TestCiv2"])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(r, AnalysisResult) for r in results.values()))

    def test_risk_breakdown_plot(self):
        analysis = self.engine.analyze_civilization("TestCiv")
        try:
            self.engine.plot_risk_breakdown(analysis)
        except Exception as e:
            self.fail(f"plot_risk_breakdown failed: {e}")

if __name__ == "__main__":
    main()
    unittest.main(argv=[''], exit=False)
