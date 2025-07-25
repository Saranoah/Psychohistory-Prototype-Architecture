# Enhanced Psychohistory Analysis Framework v2.0
# Now with temporal analysis, intervention simulation, and advanced AI impact modeling

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
from enum import Enum, auto
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

class MetricCategory(Enum):
    ECONOMIC = auto()
    SOCIAL = auto()
    POLITICAL = auto()
    ENVIRONMENTAL = auto()
    TECHNOLOGICAL = auto()
    AI_INFLUENCE = auto()

@dataclass
class Metric:
    value: float = 0.5
    weight: float = 0.2
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    trend: float = 0.0  # Rate of change (per year)
    volatility: float = 0.0  # Historical volatility

class CivilizationMetrics:
    """Enhanced metrics tracking with temporal analysis and trend detection"""
    
    def __init__(self):
        self.metrics = {
            MetricCategory.ECONOMIC: {
                'wealth_inequality': Metric(weight=0.25),
                'currency_stability': Metric(weight=0.2),
                'trade_volume': Metric(weight=0.15),
                'debt_to_gdp': Metric(weight=0.25),
                'inflation_rate': Metric(weight=0.15)
            },
            MetricCategory.SOCIAL: {
                'civic_engagement': Metric(weight=0.3),
                'social_mobility': Metric(weight=0.25),
                'population_growth': Metric(weight=0.15),
                'urbanization_rate': Metric(weight=0.1),
                'education_index': Metric(weight=0.2)
            },
            MetricCategory.POLITICAL: {
                'institutional_trust': Metric(weight=0.3),
                'corruption_index': Metric(weight=0.25),
                'political_stability': Metric(weight=0.2),
                'military_spending_ratio': Metric(weight=0.15),
                'democratic_index': Metric(weight=0.1)
            },
            MetricCategory.ENVIRONMENTAL: {
                'resource_depletion': Metric(weight=0.4),
                'climate_stress': Metric(weight=0.3),
                'agricultural_productivity': Metric(weight=0.2),
                'energy_security': Metric(weight=0.1)
            },
            MetricCategory.TECHNOLOGICAL: {
                'innovation_rate': Metric(weight=0.3),
                'information_freedom': Metric(weight=0.2),
                'digital_adoption': Metric(weight=0.2),
                'scientific_output': Metric(weight=0.3)
            },
            MetricCategory.AI_INFLUENCE: {
                'ai_penetration_rate': Metric(weight=0.2),
                'cognitive_outsourcing': Metric(weight=0.25),
                'algorithmic_governance': Metric(weight=0.2),
                'reality_authenticity_crisis': Metric(weight=0.15),
                'human_ai_symbiosis': Metric(weight=0.1),
                'ai_behavioral_conditioning': Metric(weight=0.25),
                'information_velocity': Metric(weight=0.15),
                'personalized_reality_bubbles': Metric(weight=0.2),
                'decision_dependency': Metric(weight=0.25),
                'collective_intelligence_erosion': Metric(weight=0.3)
            }
        }
        self.historical_data = []
        self.current_date = datetime.now()
    
    def update_metric(self, category: MetricCategory, name: str, value: float):
        """Update metric value and calculate trend"""
        if category in self.metrics and name in self.metrics[category]:
            metric = self.metrics[category][name]
            
            # Store previous value for trend calculation
            prev_value = metric.value
            
            # Update current value (clamped between thresholds)
            metric.value = max(metric.min_threshold, 
                              min(metric.max_threshold, value))
            
            # Calculate simple trend if we have historical data
            if self.historical_data:
                time_diff = (self.current_date - self.historical_data[-1]['date']).days / 365.25
                if time_diff > 0:
                    metric.trend = (metric.value - prev_value) / time_diff
    
    def take_snapshot(self, snapshot_date: datetime = None):
        """Record current state with timestamp"""
        if not snapshot_date:
            snapshot_date = datetime.now()
        
        snapshot = {
            'date': snapshot_date,
            'metrics': {cat.name: {k: v.value for k, v in metrics.items()} 
                      for cat, metrics in self.metrics.items()},
            'trends': {cat.name: {k: v.trend for k, v in metrics.items()} 
                      for cat, metrics in self.metrics.items()}
        }
        
        self.historical_data.append(snapshot)
        self.current_date = snapshot_date
        return snapshot

class HistoricalPattern:
    """Enhanced pattern representation with dynamic adaptation"""
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 preconditions: Dict[str, Tuple[float, float]],
                 outcome: str,
                 base_confidence: float,
                 timeframe: str,
                 severity: float = 0.7):
        self.name = name
        self.description = description
        self.preconditions = preconditions
        self.outcome = outcome
        self.base_confidence = base_confidence
        self.current_confidence = base_confidence
        self.timeframe = timeframe
        self.severity = severity
        self.historical_examples = []
        self.adaptation_factor = 1.0  # Adjusts based on modern relevance
    
    def add_example(self, civilization: str, period: str, outcome: str, modern_relevance: float = 1.0):
        """Add historical example with relevance weighting"""
        self.historical_examples.append({
            'civilization': civilization,
            'period': period,
            'outcome': outcome,
            'relevance': modern_relevance,
            'added_date': datetime.now()
        })
        self._adjust_confidence()
    
    def _adjust_confidence(self):
        """Increase confidence based on examples and relevance"""
        total_relevance = sum(ex['relevance'] for ex in self.historical_examples)
        self.current_confidence = min(0.95, 
                                    self.base_confidence + (total_relevance * 0.05))
    
    def update_for_modern_context(self, time_decay_factor: float = 0.9):
        """Adjust pattern relevance for modern contexts"""
        ancient_examples = sum(1 for ex in self.historical_examples 
                             if datetime.strptime(ex['period'][-4:], '%Y').year < 1800)
        modern_examples = len(self.historical_examples) - ancient_examples
        
        # Decrease weight for patterns with mostly ancient examples
        if modern_examples == 0 and ancient_examples > 0:
            self.adaptation_factor *= time_decay_factor

class PsychohistoryEngine:
    """Advanced analysis engine with predictive modeling"""
    
    def __init__(self):
        self.patterns = []
        self.civilizations = {}
        self.temporal_resolution = timedelta(days=90)
        self._initialize_core_patterns()
        self.risk_thresholds = {
            'LOW': 0.4,
            'MEDIUM': 0.7,
            'HIGH': 0.9
        }
    
    def _initialize_core_patterns(self):
        """Initialize with enhanced pattern set"""
        patterns = [
            {
                'name': "Economic Collapse Cycle",
                'description': "Conditions preceding economic system failure",
                'preconditions': {
                    'wealth_inequality': (0.75, 1.0),
                    'debt_to_gdp': (0.85, 1.0),
                    'currency_stability': (0.0, 0.4)
                },
                'outcome': "Currency crisis and economic breakdown",
                'base_confidence': 0.78,
                'timeframe': "3-10 years",
                'severity': 0.85
            },
            {
                'name': "Revolutionary Conditions",
                'description': "Social and political precursors to revolution",
                'preconditions': {
                    'wealth_inequality': (0.8, 1.0),
                    'social_mobility': (0.0, 0.3),
                    'institutional_trust': (0.0, 0.4)
                },
                'outcome': "Mass uprising or civil war",
                'base_confidence': 0.72,
                'timeframe': "1-7 years",
                'severity': 0.9
            },
            {
                'name': "AI Cognitive Disruption",
                'description': "Societal instability from rapid AI integration",
                'preconditions': {
                    'cognitive_outsourcing': (0.7, 1.0),
                    'reality_authenticity_crisis': (0.7, 1.0),
                    'collective_intelligence_erosion': (0.6, 1.0)
                },
                'outcome': "Cognitive disorientation and social fragmentation",
                'base_confidence': 0.65,
                'timeframe': "5-15 years",
                'severity': 0.8
            },
            {
                'name': "Techno-Authoritarian Transition",
                'description': "Shift to AI-mediated authoritarian governance",
                'preconditions': {
                    'algorithmic_governance': (0.8, 1.0),
                    'decision_dependency': (0.8, 1.0),
                    'democratic_index': (0.0, 0.3)
                },
                'outcome': "Emergence of algorithmic authoritarianism",
                'base_confidence': 0.6,
                'timeframe': "10-25 years",
                'severity': 0.95
            }
        ]
        
        for p in patterns:
            pattern = HistoricalPattern(**p)
            self.patterns.append(pattern)
        
        # Add historical examples with modern relevance factors
        self.patterns[0].add_example("Weimar Germany", "1920-1923", 
                                    "Hyperinflation crisis", 0.9)
        self.patterns[0].add_example("Roman Empire", "3rd Century", 
                                   "Currency debasement", 0.6)
        
        self.patterns[1].add_example("France", "1780-1789", 
                                    "French Revolution", 0.8)
        
        self.patterns[2].add_example("Social Media Era", "2010-2020", 
                                    "Polarization and misinformation", 0.95)
        
        self.patterns[3].add_example("China", "2010-present", 
                                    "Social credit system development", 0.85)

    def add_civilization(self, name: str, metrics: CivilizationMetrics):
        """Register a civilization with initial metrics"""
        self.civilizations[name] = {
            'metrics': metrics,
            'analyses': [],
            'risk_history': [],
            'intervention_history': []
        }
    
    def analyze_all(self, analysis_date: datetime = None):
        """Run analysis on all civilizations with parallel processing simulation"""
        if not analysis_date:
            analysis_date = datetime.now()
        
        return {name: self.analyze_civilization(name, analysis_date) 
               for name in self.civilizations}
    
    def analyze_civilization(self, civ_name: str, analysis_date: datetime):
        """Comprehensive analysis with temporal tracking"""
        if civ_name not in self.civilizations:
            raise ValueError(f"Unknown civilization: {civ_name}")
        
        civ = self.civilizations[civ_name]
        metrics = civ['metrics']
        
        # Take new snapshot if needed
        if not metrics.historical_data or \
           (analysis_date - metrics.historical_data[-1]['date']) >= self.temporal_resolution:
            metrics.take_snapshot(analysis_date)
        
        current_state = metrics.historical_data[-1]['metrics']
        current_trends = metrics.historical_data[-1]['trends']
        
        # Normalize metrics with weights and trends
        normalized = self._normalize_metrics(current_state, current_trends)
        
        # Pattern matching with trend projection
        pattern_matches = []
        for pattern in sorted(self.patterns, key=lambda x: -x.severity):
            match_score = self._calculate_match(normalized, pattern)
            if match_score >= pattern.current_confidence * 0.7:  # Adjusted threshold
                projected_impact = self._project_impact(match_score, pattern, normalized)
                pattern_matches.append({
                    'pattern': pattern.name,
                    'match_score': match_score,
                    'outcome': pattern.outcome,
                    'timeframe': pattern.timeframe,
                    'severity': pattern.severity,
                    'projected_impact': projected_impact,
                    'confidence': pattern.current_confidence
                })
        
        # Calculate composite risk score with trend projection
        risk_score = self._calculate_risk_score(normalized, pattern_matches)
        
        # Generate recommendations with projected efficacy
        recommendations = self._generate_recommendations(normalized, pattern_matches)
        
        # Store analysis
        analysis = {
            'date': analysis_date,
            'risk_score': risk_score,
            'risk_level': self._determine_risk_level(risk_score),
            'pattern_matches': pattern_matches,
            'recommendations': recommendations,
            'metric_snapshot': normalized['values'],
            'metric_trends': normalized['trends']
        }
        
        civ['analyses'].append(analysis)
        civ['risk_history'].append((analysis_date, risk_score))
        
        return analysis
    
    def _normalize_metrics(self, values: Dict, trends: Dict) -> Dict:
        """Normalize metrics with trend projection"""
        normalized = {
            'values': {},
            'trends': {},
            'projected': {}
        }
        
        # Combine all categories
        all_metrics = {}
        for cat in values:
            for metric in values[cat]:
                key = f"{cat}_{metric}"
                all_metrics[key] = {
                    'value': values[cat][metric],
                    'trend': trends[cat][metric]
                }
        
        # Normalize and project
        for metric in all_metrics:
            val = all_metrics[metric]['value']
            trend = all_metrics[metric]['trend']
            
            normalized['values'][metric] = val
            normalized['trends'][metric] = trend
            
            # Simple linear projection 1 year out
            normalized['projected'][metric] = max(0, min(1, val + trend))
        
        return normalized
    
    def _calculate_match(self, metrics: Dict, pattern: HistoricalPattern) -> float:
        """Calculate weighted pattern match with trend consideration"""
        total_weight = 0
        matched_weight = 0
        
        for metric, (min_val, max_val) in pattern.preconditions.items():
            if metric in metrics['values']:
                # Get current and projected values
                current = metrics['values'][metric]
                projected = metrics['projected'][metric]
                
                # Use weighted average of current and projected
                weighted_value = current * 0.7 + projected * 0.3
                
                # Check if within bounds
                if min_val <= weighted_value <= max_val:
                    matched_weight += 1  # Simplified for example
                total_weight += 1
        
        return matched_weight / total_weight if total_weight > 0 else 0
    
    def _project_impact(self, match_score: float, pattern: HistoricalPattern, metrics: Dict) -> float:
        """Project potential impact of matched pattern"""
        severity_factor = pattern.severity
        trend_factor = 1.0 + 0.5 * np.mean(list(metrics['trends'].values()))  # Amplify if trends worsening
        return min(1.0, match_score * severity_factor * trend_factor)
    
    def _calculate_risk_score(self, metrics: Dict, matches: List[Dict]) -> float:
        """Calculate composite risk score (0-1) with trend amplification"""
        # Base score from critical metrics
        critical_metrics = [
            'ECONOMIC_wealth_inequality',
            'POLITICAL_institutional_trust',
            'SOCIAL_civic_engagement',
            'AI_INFLUENCE_collective_intelligence_erosion'
        ]
        
        base_score = np.mean([metrics['values'].get(m, 0.5) for m in critical_metrics])
        
        # Trend amplification factor
        trend_factor = 1.0 + 0.5 * np.mean([t for t in metrics['trends'].values() if t > 0])
        
        # Pattern impact
        pattern_impact = max([m['projected_impact'] for m in matches], default=0)
        
        return min(1.0, (base_score * 0.5 + pattern_impact * 0.5) * trend_factor)
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level based on thresholds"""
        if score >= self.risk_thresholds['HIGH']:
            return "CRITICAL" if score > 0.95 else "HIGH"
        elif score >= self.risk_thresholds['MEDIUM']:
            return "MEDIUM"
        elif score >= self.risk_thresholds['LOW']:
            return "LOW"
        return "STABLE"
    
    def _generate_recommendations(self, metrics: Dict, matches: List[Dict]) -> List[Dict]:
        """Generate targeted recommendations with projected efficacy"""
        recs = []
        
        # Critical metric recommendations
        critical_metrics = {
            'ECONOMIC_wealth_inequality': {
                'threshold': 0.7,
                'recommendations': [
                    ("Implement progressive wealth taxation", 0.7),
                    ("Universal basic asset programs", 0.6)
                ]
            },
            'POLITICAL_institutional_trust': {
                'threshold': 0.4,
                'recommendations': [
                    ("Transparency initiatives for government", 0.65),
                    ("Citizen oversight boards", 0.55)
                ]
            },
            'AI_INFLUENCE_collective_intelligence_erosion': {
                'threshold': 0.6,
                'recommendations': [
                    ("Cognitive independence education", 0.7),
                    ("Regulate algorithmic conditioning", 0.6),
                    ("Human-only decision spaces", 0.5)
                ]
            }
        }
        
        for metric, data in critical_metrics.items():
            if metrics['values'].get(metric, 0) > data['threshold']:
                for text, efficacy in data['recommendations']:
                    # Adjust efficacy based on trends
                    trend = metrics['trends'].get(metric, 0)
                    adjusted_efficacy = efficacy * (1 - 0.3 * trend)  # Reduce efficacy if trend worsening
                    recs.append({
                        'action': text,
                        'category': metric.split('_')[0],
                        'efficacy': adjusted_efficacy,
                        'time_horizon': "short" if adjusted_efficacy > 0.6 else "medium"
                    })
        
        # Pattern-specific recommendations
        pattern_recs = {
            "Economic Collapse Cycle": [
                ("Currency stabilization fund", 0.75),
                ("Debt restructuring program", 0.7)
            ],
            "AI Cognitive Disruption": [
                ("Authenticity verification standards", 0.8),
                ("AI literacy curriculum", 0.65)
            ],
            "Techno-Authoritarian Transition": [
                ("Algorithmic accountability laws", 0.7),
                ("Human oversight requirements", 0.6)
            ]
        }
        
        for match in matches:
            if match['pattern'] in pattern_recs:
                for text, efficacy in pattern_recs[match['pattern']]:
                    # Adjust based on pattern severity
                    adjusted_efficacy = efficacy * (1 - 0.1 * match['severity'])
                    recs.append({
                        'action': text,
                        'category': "PATTERN_" + match['pattern'],
                        'efficacy': adjusted_efficacy,
                        'time_horizon': "urgent" if match['severity'] > 0.8 else "medium"
                    })
        
        # Deduplicate and sort by efficacy
        unique_recs = {}
        for rec in recs:
            key = rec['action']
            if key not in unique_recs or rec['efficacy'] > unique_recs[key]['efficacy']:
                unique_recs[key] = rec
        
        return sorted(unique_recs.values(), key=lambda x: -x['efficacy'])
    
    def simulate_intervention(self, civ_name: str, intervention: Dict[str, float], 
                            years: int = 5, persist: bool = False) -> Dict:
        """Simulate policy interventions with realistic delayed effects"""
        if civ_name not in self.civilizations:
            raise ValueError(f"Unknown civilization: {civ_name}")
        
        civ = self.civilizations[civ_name]
        current = civ['metrics'].historical_data[-1]
        
        # Create projected metrics with intervention effects
        projected = {cat: metrics.copy() for cat, metrics in current['metrics'].items()}
        
        # Apply interventions with diminishing returns
        for metric_change, effect_size in intervention.items():
            category, metric = metric_change.split('_', 1)
            if category in projected and metric in projected[category]:
                current_val = projected[category][metric]
                
                # Diminishing returns formula
                adjusted_effect = effect_size * (1 - current_val * 0.5)
                
                # Apply over time with some randomness
                projected[category][metric] = max(0, min(1, 
                    current_val + adjusted_effect * (1 - 0.1 * np.random.random())))
        
        # Create temporary civilization for projection
        temp_metrics = CivilizationMetrics()
        for cat in projected:
            for metric, value in projected[cat].items():
                temp_metrics.update_metric(MetricCategory[cat], metric, value)
        
        # Store intervention record
        intervention_record = {
            'date': current['date'],
            'intervention': intervention,
            'projected_years': years,
            'applied': persist
        }
        
        if persist:
            civ['intervention_history'].append(intervention_record)
        
        # Analyze projected state
        temp_engine = PsychohistoryEngine()
        temp_engine.add_civilization("projection", temp_metrics)
        return temp_engine.analyze_civilization("projection", 
                                              current['date'] + timedelta(days=365*years))
    
    def visualize_risk_trends(self, civ_name: str, show_interventions: bool = True):
        """Generate enhanced risk visualization with interventions"""
        if civ_name not in self.civilizations:
            raise ValueError(f"Unknown civilization: {civ_name}")
        
        civ = self.civilizations[civ_name]
        if not civ['risk_history']:
            raise ValueError("No historical data available")
        
        dates, scores = zip(*civ['risk_history'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, scores, marker='o', linestyle='-', label='Risk Score')
        
        # Add risk thresholds
        for level, threshold in self.risk_thresholds.items():
            plt.axhline(y=threshold, color='red' if level == 'HIGH' else 'orange' if level == 'MEDIUM' else 'yellow',
                       linestyle='--', alpha=0.5)
            plt.text(dates[-1], threshold + 0.02, f"{level} Risk", 
                    color='red' if level == 'HIGH' else 'orange' if level == 'MEDIUM' else 'yellow')
        
        # Add interventions if available
        if show_interventions and civ['intervention_history']:
            for intervention in civ['intervention_history']:
                plt.axvline(x=intervention['date'], color='green', linestyle=':', alpha=0.7)
                plt.text(intervention['date'], 0.1, 
                        '\n'.join([f"{k}: {v:.2f}" for k, v in intervention['intervention'].items()]),
                        rotation=90, va='bottom', color='green')
        
        plt.title(f"Risk Score Trend for {civ_name}")
        plt.xlabel("Date")
        plt.ylabel("Composite Risk Score (0-1)")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def get_critical_metrics(self, civ_name: str, n: int = 5) -> List[Tuple[str, float]]:
        """Identify metrics contributing most to risk score"""
        analysis = self.analyze_civilization(civ_name, datetime.now())
        metrics = analysis['metric_snapshot']
        
        # Calculate deviation from ideal (0.5 is neutral)
        deviations = {k: abs(v - 0.5) for k, v in metrics.items()}
        
        # Get top n metrics with highest deviation
        return sorted(deviations.items(), key=lambda x: -x[1])[:n]

# Example Usage
if __name__ == "__main__":
    # Initialize engine
    engine = PsychohistoryEngine()
    
    # Create and configure a civilization
    usa = CivilizationMetrics()
    
    # Set concerning metrics (hypothetical modern state)
    usa.update_metric(MetricCategory.ECONOMIC, 'wealth_inequality', 0.82)
    usa.update_metric(MetricCategory.ECONOMIC, 'debt_to_gdp', 0.95)
    usa.update_metric(MetricCategory.POLITICAL, 'institutional_trust', 0.35)
    usa.update_metric(MetricCategory.SOCIAL, 'social_mobility', 0.38)
    usa.update_metric(MetricCategory.POLITICAL, 'corruption_index', 0.55)
    
    # Set AI influence metrics
    usa.update_metric(MetricCategory.AI_INFLUENCE, 'cognitive_outsourcing', 0.68)
    usa.update_metric(MetricCategory.AI_INFLUENCE, 'reality_authenticity_crisis', 0.72)
    usa.update_metric(MetricCategory.AI_INFLUENCE, 'collective_intelligence_erosion', 0.65)
    
    # Add to engine
    engine.add_civilization("United States", usa)
    
    # Run analysis
    analysis = engine.analyze_civilization("United States", datetime.now())
    
    print("\nPsychohistory Analysis Report")
    print("="*50)
    print(f"Civilization: United States")
    print(f"Analysis Date: {analysis['date']}")
    print(f"\nRisk Level: {analysis['risk_level']} ({analysis['risk_score']:.2f}/1.0)")
    
    if analysis['pattern_matches']:
        print("\n⚠️ Critical Pattern Matches:")
        for match in analysis['pattern_matches']:
            print(f"- {match['pattern']} ({match['match_score']:.0%} match)")
            print(f"  Outcome: {match['outcome']}")
            print(f"  Timeframe: {match['timeframe']}")
            print(f"  Projected Impact: {match['projected_impact']:.2f}")
    
    print("\nTop Critical Metrics:")
    for metric, deviation in engine.get_critical_metrics("United States"):
        print(f"- {metric}: {analysis['metric_snapshot'][metric]:.2f} "
              f"(Trend: {analysis['metric_trends'][metric]:+.2f}/yr)")
    
    print("\nRecommended Interventions (by efficacy):")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"{i}. {rec['action']} (Eff: {rec['efficacy']:.0%}, Horizon: {rec['time_horizon']})")
    
    # Simulate intervention
    print("\nSimulating Wealth Inequality Reduction...")
    sim_result = engine.simulate_intervention(
        "United States",
        {"ECONOMIC_wealth_inequality": -0.2,
         "SOCIAL_social_mobility": 0.15},
        years=10,
        persist=True
    )
    print(f"Projected 10-year risk score: {sim_result['risk_score']:.2f} "
          f"(Change: {sim_result['risk_score'] - analysis['risk_score']:+.2f})")
    
    # Generate visualization
    engine.visualize_risk_trends("United States")
