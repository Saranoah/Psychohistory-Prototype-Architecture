import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import logging
import matplotlib.pyplot as plt
import unittest
from copy import copy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        """Return weight for a metric in pattern matching."""
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
            )
        ]

    def get_active_patterns(self) -> List[HistoricalPattern]:
        return self.patterns

    def get_patterns_by_timeframe(self, timeframe: str) -> List[HistoricalPattern]:
        return [p for p in self.patterns if p.timeframe == timeframe]

class CivilizationMetrics:
    def __init__(self):
        self.historical_data = []
        self.current_state = {cat.value: {} for cat in MetricCategory}
        self.metric_ranges = {cat.value: {} for cat in MetricCategory}

    def take_snapshot(self, date: datetime):
        """Store a snapshot of current metrics and trends."""
        if not any(self.current_state.values()):
            raise ValueError("No metrics available for snapshot")
        self.historical_data.append({
            'date': date,
            'metrics': {cat: dict(metrics) for cat, metrics in self.current_state.items()},
            'trends': self._calculate_trends()
        })

    def _calculate_trends(self) -> Dict:
        """Calculate trends based on historical data."""
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
        """Efficiently clone metrics."""
        new_obj = CivilizationMetrics()
        new_obj.current_state = {cat: dict(metrics) for cat, metrics in self.current_state.items()}
        new_obj.metric_ranges = {cat: dict(ranges) for cat, ranges in self.metric_ranges.items()}
        new_obj.historical_data = [dict(d) for d in self.historical_data]
        return new_obj

    def update_metric(self, category: MetricCategory, metric: str, value: float, min_val: float = 0.0, max_val: float = 1.0):
        """Update a metric with normalization and validation."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"Metric value for {metric} must be numeric, got {type(value)}")
        normalized_value = MetricNormalizer.normalize(value, min_val, max_val)
        self.current_state[category.value][metric] = normalized_value
        self.metric_ranges[category.value][metric] = (min_val, max_val)

    def get_metric(self, category: MetricCategory, metric: str) -> float:
        return self.current_state.get(category.value, {}).get(metric, 0.0)

class MetricNormalizer:
    @staticmethod
    def normalize(value: float, min_val: float, max_val: float) -> float:
        if max_val == min_val:
            logging.warning(f"Min and max values are equal for normalization: {min_val}")
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    @staticmethod
    def denormalize(normalized: float, min_val: float, max_val: float) -> float:
        return min_val + normalized * (max_val - min_val)

class UncertaintyQuantifier:
    def analyze_uncertainty(self, metrics: Dict, patterns: List[Dict], risk_score: float) -> Dict:
        """Quantify uncertainty using Monte Carlo simulation."""
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
        base_risk = np.mean([v for k, v in metrics.items() if 'inequality' in k or 'stress' in k], initial=0.5)
        pattern_risk = max((p['severity'] * p['match_score'] for p in patterns), default=0.0)
        return min(1.0, 0.6 * base_risk + 0.4 * pattern_risk)

    def _compute_sensitivity(self, metrics: Dict, patterns: List[Dict]) -> Dict:
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

class PredictionModel:
    @staticmethod
    def linear_projection(current: float, trend: float, years: int, decay: float = 0.7) -> float:
        effective_trend = trend * (decay ** years)
        projected = current + (effective_trend * years)
        return max(0.0, min(1.0, projected))

    @staticmethod
    def logistic_growth(current: float, capacity: float, growth_rate: float, years: int) -> float:
        if growth_rate == 0:
            return current
        exponent = -growth_rate * years
        return capacity / (1 + ((capacity - current) / current) * np.exp(exponent))

    @staticmethod
    def momentum_weighted(current: float, trends: List[float], years: int) -> float:
        if not trends:
            return current
        weights = np.array([0.5 ** i for i in range(len(trends))])
        weights = weights / weights.sum()
        avg_trend = np.average(trends, weights=weights)
        return max(0.0, min(1.0, current + (avg_trend * years)))

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
    """
    def __init__(self, config_class=EngineConfig):
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
        self._normalize_cache = {}

    def add_civilization(self, name: str, metrics: CivilizationMetrics) -> None:
        """Register a civilization with validation."""
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
        """Comprehensive analysis with uncertainty quantification."""
        self._validate_civilization(civ_name)
        if analysis_date is None:
            analysis_date = datetime.now()
        civ = self.civilizations[civ_name]
        result = self._run_analysis_logic(civ['metrics'], analysis_date)
        civ['analyses'].append(result)
        civ['risk_history'].append((analysis_date, result.risk_score))
        return result

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
        """Prepare metrics for analysis."""
        if (not metrics.historical_data or 
            (analysis_date - metrics.historical_data[-1]['date']) >= self.temporal_resolution):
            metrics.take_snapshot(analysis_date)
        if not metrics.historical_data:
            raise ValueError("No historical data available for analysis")
        return metrics

    def predict_timeline(self, civ_name: str, years: int = 5, include_interventions: List[Dict] = None) -> List[TimelinePrediction]:
        """Accurate multi-year prediction with trend propagation and intervention modeling."""
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
        """Project all metrics forward using appropriate models."""
        projected_metrics = metrics.clone()
        current_state = projected_metrics.historical_data[-1]['metrics'] if projected_metrics.historical_data else projected_metrics.current_state
        current_trends = projected_metrics.historical_data[-1].get('trends', {}) if projected_metrics.historical_data else {}

        keys, values, trends = [], [], []
        for category, metric_dict in current_state.items():
            for metric_name, value in metric_dict.items():
                keys.append((category, metric_name))
                values.append(value)
                trends.append(current_trends.get(category, {}).get(metric_name, 0.0))

        values = np.array(values)
        trends = np.array(trends)
        projected_values = np.zeros_like(values)

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

        for (category, metric_name), value in zip(keys, projected_values):
            projected_metrics.update_metric(MetricCategory[category.upper()], metric_name, value)

        projected_date = datetime.now() + timedelta(days=365 * years)
        projected_metrics.take_snapshot(projected_date)
        return projected_metrics

    def _get_historical_trends(self, metrics: CivilizationMetrics, metric_key: str) -> List[float]:
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
        if '_' in metric_key:
            category, metric_name = metric_key.split('_', 1)
            return state.get(category, {}).get(metric_name)
        return None

    def simulate_intervention(self, civ_name: str, intervention: Dict[str, float], years: int = 5) -> InterventionResult:
        """Realistic intervention simulation with diminishing returns and side effects."""
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
        total_magnitude = sum(abs(v) for v in intervention.values())
        risk_factor = 1.0 - (current_risk * 0.5)
        base_prob = min(0.9, total_magnitude * 2.0)
        return max(0.1, base_prob * risk_factor)

    def _calculate_cost_benefit(self, intervention: Dict, effectiveness: float) -> float:
        total_cost = sum(abs(v) for v in intervention.values())
        return float('inf') if total_cost == 0 else effectiveness / total_cost

    def _generate_year_predictions(self, analysis: AnalysisResult, year: int) -> List[Dict]:
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
        critical_metrics = []
        for metric, value in analysis.metric_snapshot.items():
            if ('inequality' in metric or 'stress' in metric) and value > 0.7:
                critical_metrics.append(metric)
            elif ('trust' in metric or 'engagement' in metric) and value < 0.3:
                critical_metrics.append(metric)
        return critical_metrics

    def _identify_stability_factors(self, analysis: AnalysisResult) -> List[str]:
        stability_metrics = []
        for metric, value in analysis.metric_snapshot.items():
            if ('stability' in metric or 'growth' in metric) and value > 0.6:
                stability_metrics.append(metric)
            elif ('trust' in metric or 'engagement' in metric) and value > 0.6:
                stability_metrics.append(metric)
        return stability_metrics

    def _validate_civilization(self, civ_name: str) -> None:
        if civ_name not in self.civilizations:
            logging.error(f"Unknown civilization: {civ_name}")
            raise ValueError(f"Unknown civilization: {civ_name}")

    def _normalize_metrics(self, values: Dict, trends: Dict) -> Dict:
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
                    projected = (value * c_w) + ((value + trend) * p_w)
                    normalized['projected'][key] = max(0.0, min(1.0, projected))
                except (TypeError, ValueError) as e:
                    logging.warning(f"Invalid metric {key}: {e}")
                    continue
        self._normalize_cache[cache_key] = normalized
        return normalized

    def _match_patterns(self, metrics: Dict) -> List[Dict]:
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
        total_weight = 0.0
        matched_weight = 0.0
        c_w = self.config.PROJECTION_WEIGHTS['current']
        p_w = self.config.PROJECTION_WEIGHTS['projected']
        for metric_key, (min_val, max_val) in pattern.preconditions.items():
            if metric_key not in metrics['values']:
                logging.warning(f"Metric {metric_key} not found in metrics")
                continue
            weight = pattern.get_metric_weight(metric_key)
            total_weight += weight
            current_value = metrics['values'][metric_key]
            projected_value = metrics['projected'][metric_key]
            effective_value = current_value * c_w + projected_value * p_w
            if min_val <= effective_value <= max_val:
                matched_weight += weight
            else:
                distance = min_val - effective_value if effective_value < min_val else effective_value - max_val
                partial_match = np.exp(-distance * self.config.PARTIAL_MATCH_DECAY)
                matched_weight += weight * partial_match
        return matched_weight / total_weight if total_weight > 0 else 0.0

    def _calculate_composite_risk(self, metrics: Dict, pattern_matches: List[Dict]) -> float:
        critical_metrics = [
            'ECONOMIC_wealth_inequality', 'POLITICAL_institutional_trust',
            'SOCIAL_civic_engagement', 'ENVIRONMENTAL_climate_stress'
        ]
        metric_risks = []
        for metric in critical_metrics:
            if metric in metrics['values']:
                value = metrics['values'][metric]
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
        recommendations = {
            "Economic Collapse": "Implement progressive taxation and social safety nets",
            "Political Instability": "Strengthen democratic institutions and public communication"
        }
        return recommendations.get(pattern_name, "Review and address underlying metrics")

# ==================== USAGE EXAMPLE AND TESTING ====================

def main():
    """Demonstrate the Psychohistory Engine with realistic data and visualization."""
    print("=== Psychohistory Engine - Production Ready ===")
    engine = PsychohistoryEngine()
    metrics = CivilizationMetrics()

    # Initialize realistic metrics
    metrics.update_metric(MetricCategory.ECONOMIC, "wealth_inequality", 70, 0, 100)
    metrics.update_metric(MetricCategory.ECONOMIC, "economic_stability", 30, 0, 100)
    metrics.update_metric(MetricCategory.POLITICAL, "institutional_trust", 40, 0, 100)
    metrics.update_metric(MetricCategory.SOCIAL, "civic_engagement", 50, 0, 100)
    metrics.update_metric(MetricCategory.ENVIRONMENTAL, "climate_stress", 60, 0, 100)
    metrics.take_snapshot(datetime.now())
    engine.add_civilization("Earth2025", metrics)

    # Run analysis
    logging.info("Running Analysis...")
    analysis = engine.analyze_civilization("Earth2025")
    print(f"Risk Score: {analysis.risk_score:.3f} ({analysis.risk_level})")
    print(f"Pattern Matches: {len(analysis.pattern_matches)}")
    print("Recommendations:", [r['action'] for r in analysis.recommendations])

    # Generate timeline predictions
    logging.info("Generating Timeline Predictions...")
    timeline = engine.predict_timeline("Earth2025", years=3)
    years = [p.year for p in timeline]
    risks = [p.risk_score for p in timeline]
    plt.plot(years, risks, marker='o')
    plt.title("Risk Score Over Time")
    plt.xlabel("Years")
    plt.ylabel("Risk Score")
    plt.grid(True)
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

if __name__ == "__main__":
    main()
    unittest.main(argv=[''], exit=False)
