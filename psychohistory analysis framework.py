# Enhanced Psychohistory Analysis Framework
# Now with temporal analysis, cross-civilization comparison, and intervention simulation

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
from enum import Enum
import matplotlib.pyplot as plt

class MetricCategory(Enum):
    ECONOMIC = "economic"
    SOCIAL = "social"
    POLITICAL = "political"
    ENVIRONMENTAL = "environmental"
    TECHNOLOGICAL = "technological"

class CivilizationMetrics:
    """Enhanced metrics tracking with temporal dimension"""
    
    def __init__(self):
        self.metrics = {
            MetricCategory.ECONOMIC: {
                'wealth_inequality': {'value': 0.5, 'weight': 0.25},
                'currency_stability': {'value': 0.5, 'weight': 0.2},
                'trade_volume': {'value': 0.5, 'weight': 0.15},
                'debt_to_gdp': {'value': 0.5, 'weight': 0.25},
                'inflation_rate': {'value': 0.5, 'weight': 0.15}
            },
            MetricCategory.SOCIAL: {
                'civic_engagement': {'value': 0.5, 'weight': 0.3},
                'social_mobility': {'value': 0.5, 'weight': 0.25},
                'population_growth': {'value': 0.5, 'weight': 0.15},
                'urbanization_rate': {'value': 0.5, 'weight': 0.1},
                'education_index': {'value': 0.5, 'weight': 0.2}
            },
            MetricCategory.POLITICAL: {
                'institutional_trust': {'value': 0.5, 'weight': 0.3},
                'corruption_index': {'value': 0.5, 'weight': 0.25},
                'political_stability': {'value': 0.5, 'weight': 0.2},
                'military_spending_ratio': {'value': 0.5, 'weight': 0.15},
                'democratic_index': {'value': 0.5, 'weight': 0.1}
            },
            MetricCategory.ENVIRONMENTAL: {
                'resource_depletion': {'value': 0.5, 'weight': 0.4},
                'climate_stress': {'value': 0.5, 'weight': 0.3},
                'agricultural_productivity': {'value': 0.5, 'weight': 0.2},
                'energy_security': {'value': 0.5, 'weight': 0.1}
            },
            MetricCategory.TECHNOLOGICAL: {
                'innovation_rate': {'value': 0.5, 'weight': 0.3},
                'information_freedom': {'value': 0.5, 'weight': 0.2},
                'digital_adoption': {'value': 0.5, 'weight': 0.2},
                'scientific_output': {'value': 0.5, 'weight': 0.3}
            }
        }
        self.historical_data = []
        self.current_snapshot_date = datetime.now()
    
    def update_metric(self, category: MetricCategory, metric_name: str, value: float):
        if category in self.metrics and metric_name in self.metrics[category]:
            self.metrics[category][metric_name]['value'] = max(0.0, min(1.0, value))
        else:
            print(f"Warning: Unknown metric {category.value}_{metric_name}")
    
    def get_metric_weight(self, category: MetricCategory, metric_name: str) -> float:
        """Safely get metric weight"""
        if category in self.metrics and metric_name in self.metrics[category]:
            return self.metrics[category][metric_name].get('weight', 0.2)
        return 0.0
    
    def take_snapshot(self, snapshot_date: datetime = None):
        """Record current state with timestamp"""
        if not snapshot_date:
            snapshot_date = datetime.now()
        
        snapshot = {
            'date': snapshot_date,
            'metrics': {cat.value: {k: v['value'] for k, v in metrics.items()} 
                        for cat, metrics in self.metrics.items()}
        }
        self.historical_data.append(snapshot)
        self.current_snapshot_date = snapshot_date
        return snapshot

class HistoricalPattern:
    """Enhanced pattern representation with dynamic weighting"""
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 preconditions: Dict[str, Tuple[float, float]],
                 outcome: str,
                 confidence: float,
                 timeframe: str,
                 severity: float = 0.7):
        self.name = name
        self.description = description
        self.preconditions = preconditions
        self.outcome = outcome
        self.confidence = confidence
        self.timeframe = timeframe
        self.severity = severity
        self.historical_examples = []
        self.dynamic_weight = 1.0  # Adjusts based on recent matches
    
    def add_example(self, civilization: str, period: str, outcome: str):
        self.historical_examples.append({
            'civilization': civilization,
            'period': period,
            'outcome': outcome,
            'added_date': datetime.now()
        })
        self._adjust_confidence()
    
    def _adjust_confidence(self):
        """Increase confidence slightly with more examples"""
        self.confidence = min(0.95, self.confidence * 1.05)

class PsychohistoryEngine:
    """Enhanced analysis engine with temporal tracking and intervention simulation"""
    
    def __init__(self):
        self.patterns = []
        self.civilizations = {}
        self.temporal_resolution = timedelta(days=90)  # Default analysis frequency
        self._initialize_core_patterns()
    
    def _initialize_core_patterns(self):
        # Core patterns with refined thresholds
        patterns = [
            {
                'name': "Economic Collapse Cycle",
                'description': "Conditions preceding economic system failure",
                'preconditions': {
                    'economic_wealth_inequality': (0.75, 1.0),
                    'economic_debt_to_gdp': (0.85, 1.0),
                    'economic_currency_stability': (0.0, 0.4)
                },
                'outcome': "Currency crisis and economic breakdown",
                'confidence': 0.78,
                'timeframe': "3-10 years",
                'severity': 0.85
            },
            {
                'name': "Revolutionary Conditions",
                'description': "Social and political precursors to revolution",
                'preconditions': {
                    'economic_wealth_inequality': (0.8, 1.0),
                    'social_social_mobility': (0.0, 0.3),
                    'political_institutional_trust': (0.0, 0.4)
                },
                'outcome': "Mass uprising or civil war",
                'confidence': 0.72,
                'timeframe': "1-7 years",
                'severity': 0.9
            },
            {
                'name': "Imperial Decline",
                'description': "Signs of hegemonic power collapse",
                'preconditions': {
                    'political_military_spending_ratio': (0.7, 1.0),
                    'economic_debt_to_gdp': (0.7, 1.0),
                    'political_political_stability': (0.0, 0.5)
                },
                'outcome': "Gradual loss of global influence",
                'confidence': 0.8,
                'timeframe': "10-50 years",
                'severity': 0.75
            }
        ]
        
        for p in patterns:
            pattern = HistoricalPattern(**p)
            self.patterns.append(pattern)
        
        # Add historical examples
        self.patterns[0].add_example("Weimar Germany", "1920-1923", "Hyperinflation crisis")
        self.patterns[1].add_example("France", "1780-1789", "French Revolution")
        self.patterns[2].add_example("British Empire", "1914-1945", "Post-war decline")

    def add_civilization(self, name: str, metrics: CivilizationMetrics):
        """Register a civilization with initial metrics"""
        self.civilizations[name] = {
            'metrics': metrics,
            'analyses': [],
            'risk_history': []
        }
    
    def analyze_all(self, analysis_date: datetime = None):
        """Run analysis on all registered civilizations"""
        if not analysis_date:
            analysis_date = datetime.now()
        
        results = {}
        for civ_name in self.civilizations:
            results[civ_name] = self.analyze_civilization(civ_name, analysis_date)
        
        return results
    
    def analyze_civilization(self, civ_name: str, analysis_date: datetime = None):
        """Comprehensive analysis of a civilization's state"""
        if civ_name not in self.civilizations:
            raise ValueError(f"Unknown civilization: {civ_name}")
        
        if analysis_date is None:
            analysis_date = datetime.now()
        
        civ = self.civilizations[civ_name]
        metrics = civ['metrics']
        
        # Take new snapshot if needed
        if not metrics.historical_data or \
           (analysis_date - metrics.historical_data[-1]['date']) >= self.temporal_resolution:
            metrics.take_snapshot(analysis_date)
        
        current_state = metrics.historical_data[-1]['metrics']
        
        # Normalize metrics with weights
        normalized = self._normalize_metrics(current_state, metrics)
        
        # Pattern matching
        pattern_matches = []
        for pattern in self.patterns:
            match_score = self._calculate_match(normalized, pattern, metrics)
            if match_score >= pattern.confidence * 0.8:  # Slightly relaxed threshold
                pattern_matches.append({
                    'pattern': pattern.name,
                    'match_score': match_score,
                    'outcome': pattern.outcome,
                    'timeframe': pattern.timeframe,
                    'severity': pattern.severity,
                    'confidence': pattern.confidence
                })
        
        # Calculate composite risk score
        risk_score = self._calculate_risk_score(normalized, pattern_matches)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(normalized, pattern_matches)
        
        # Store analysis
        analysis = {
            'date': analysis_date,
            'risk_score': risk_score,
            'pattern_matches': pattern_matches,
            'recommendations': recommendations,
            'metric_snapshot': normalized
        }
        
        civ['analyses'].append(analysis)
        civ['risk_history'].append((analysis_date, risk_score))
        
        return analysis
    
    def _normalize_metrics(self, metrics: Dict[str, Dict[str, float]], 
                          civ_metrics: CivilizationMetrics) -> Dict[str, float]:
        """Apply category weights and normalize to 0-1 scale"""
        normalized = {}
        for category_str, metrics_dict in metrics.items():
            for metric, value in metrics_dict.items():
                key = f"{category_str}_{metric}"
                normalized[key] = value
        return normalized
    
    def _calculate_match(self, metrics: Dict[str, float], pattern: HistoricalPattern, 
                        civ_metrics: CivilizationMetrics) -> float:
        """Calculate weighted pattern match score - FIXED VERSION"""
        total_weight = 0
        matched_weight = 0
        
        for metric_key, (min_val, max_val) in pattern.preconditions.items():
            if metric_key in metrics:
                # Parse category and metric name correctly
                try:
                    parts = metric_key.split('_')
                    if len(parts) >= 2:
                        category_str = parts[0]
                        metric_name = '_'.join(parts[1:])  # Handle metrics with underscores
                    else:
                        raise ValueError("Invalid metric format")
                    # Find the corresponding MetricCategory enum
                    category_enum = None
                    for cat_enum in MetricCategory:
                        if cat_enum.value == category_str:
                            category_enum = cat_enum
                            break
                    
                    if category_enum:
                        weight = civ_metrics.get_metric_weight(category_enum, metric_name)
                    else:
                        weight = 0.2  # Default weight
                        
                except ValueError:
                    # If splitting fails, use default weight
                    weight = 0.2
                
                total_weight += weight
                
                if min_val <= metrics[metric_key] <= max_val:
                    matched_weight += weight
        
        return matched_weight / total_weight if total_weight > 0 else 0
    
    def _calculate_risk_score(self, metrics: Dict[str, float], matches: List[Dict]) -> float:
        """Calculate composite risk score (0-1)"""
        # Base score from metrics
        critical_metrics = [
            'economic_wealth_inequality',
            'political_institutional_trust',
            'social_civic_engagement',
            'environmental_climate_stress'
        ]
        
        available_metrics = [metrics.get(m, 0.5) for m in critical_metrics if m in metrics]
        if available_metrics:
            base_score = np.mean(available_metrics)
        else:
            base_score = 0.5  # Neutral if no metrics available
        
        # Adjust for pattern matches
        match_adjustment = 0
        if matches:
            worst_match = max(matches, key=lambda x: x['severity'])
            match_adjustment = worst_match['severity'] * worst_match['match_score']
        
        return min(1.0, base_score * 0.6 + match_adjustment * 0.4)
    
    def _generate_recommendations(self, metrics: Dict[str, float], matches: List[Dict]) -> List[str]:
        """Generate targeted recommendations"""
        recs = []
        
        # Economic recommendations
        if metrics.get('economic_wealth_inequality', 0) > 0.7:
            recs.append("Implement progressive wealth taxation to reduce inequality")
        if metrics.get('economic_debt_to_gdp', 0) > 0.8:
            recs.append("Establish debt reduction plan with spending caps")
        
        # Political recommendations
        if metrics.get('political_institutional_trust', 1) < 0.4:
            recs.append("Launch transparency initiatives for government institutions")
        if metrics.get('political_corruption_index', 0) > 0.6:
            recs.append("Strengthen independent anti-corruption mechanisms")
        
        # Add pattern-specific recommendations
        for match in matches:
            if "Economic Collapse" in match['pattern']:
                recs.append("Establish currency stabilization fund")
                recs.append("Diversify economic base to reduce systemic risk")
            elif "Revolutionary" in match['pattern']:
                recs.append("Create social mobility programs for disadvantaged groups")
                recs.append("Establish civic dialogue forums")
        
        return list(set(recs))  # Remove duplicates
    
    def simulate_intervention(self, civ_name: str, intervention: Dict[str, float], years: int = 5):
        """Simulate the impact of policy interventions over time"""
        if civ_name not in self.civilizations:
            raise ValueError(f"Unknown civilization: {civ_name}")
        
        civ = self.civilizations[civ_name]
        if not civ['metrics'].historical_data:
            # Take snapshot if none exists
            civ['metrics'].take_snapshot()
            
        current = civ['metrics'].historical_data[-1]['metrics']
        
        # Create projected metrics
        projected = {cat: metrics.copy() for cat, metrics in current.items()}
        
        # Apply interventions
        for metric_change, effect_size in intervention.items():
            try:
                parts = metric_change.split('_')
                if len(parts) >= 2:
                    category_str = parts[0]
                    metric = '_'.join(parts[1:])  # Handle metrics with underscores
                else:
                    raise ValueError("Invalid format")
                    
                if category_str in projected and metric in projected[category_str]:
                    projected[category_str][metric] = max(0, min(1, 
                        projected[category_str][metric] + effect_size))
            except ValueError:
                print(f"Warning: Invalid metric format: {metric_change}")
        
        # Create temporary civilization for projection
        temp_metrics = CivilizationMetrics()
        for cat_str in projected:
            for metric, value in projected[cat_str].items():
                # Find the corresponding enum
                for cat_enum in MetricCategory:
                    if cat_enum.value == cat_str:
                        temp_metrics.update_metric(cat_enum, metric, value)
                        break
        
        temp_metrics.take_snapshot(datetime.now() + timedelta(days=365*years))
        
        # Analyze projected state
        temp_engine = PsychohistoryEngine()
        temp_engine.add_civilization("projection", temp_metrics)
        return temp_engine.analyze_civilization("projection", 
                                              datetime.now() + timedelta(days=365*years))
    
    def visualize_risk_trends(self, civ_name: str):
        """Generate risk score visualization over time"""
        if civ_name not in self.civilizations:
            raise ValueError(f"Unknown civilization: {civ_name}")
        
        history = self.civilizations[civ_name]['risk_history']
        if not history:
            raise ValueError("No historical data available")
        
        dates, scores = zip(*history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(dates, scores, marker='o', linestyle='-')
        plt.title(f"Risk Score Trend for {civ_name}")
        plt.xlabel("Date")
        plt.ylabel("Composite Risk Score (0-1)")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Initialize engine
    engine = PsychohistoryEngine()
    
    # Create and configure a civilization
    usa = CivilizationMetrics()
    
    # Set some concerning metrics (hypothetical)
    usa.update_metric(MetricCategory.ECONOMIC, 'wealth_inequality', 0.82)
    usa.update_metric(MetricCategory.ECONOMIC, 'debt_to_gdp', 0.95)
    usa.update_metric(MetricCategory.POLITICAL, 'institutional_trust', 0.35)
    usa.update_metric(MetricCategory.SOCIAL, 'social_mobility', 0.38)
    usa.update_metric(MetricCategory.POLITICAL, 'corruption_index', 0.55)
    
    # Add to engine
    engine.add_civilization("United States", usa)
    
    # Run analysis
    analysis = engine.analyze_civilization("United States", datetime.now())
    
    print("\nPsychohistory Analysis Report")
    print("="*50)
    print(f"Civilization: United States")
    print(f"Analysis Date: {analysis['date']}")
    print(f"\nComposite Risk Score: {analysis['risk_score']:.2f}/1.0")
    
    if analysis['pattern_matches']:
        print("\n⚠️ Pattern Matches Detected:")
        for match in analysis['pattern_matches']:
            print(f"- {match['pattern']} ({match['match_score']:.0%} match)")
            print(f"  Potential Outcome: {match['outcome']}")
            print(f"  Timeframe: {match['timeframe']}")
    
    print("\nRecommended Interventions:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Simulate an intervention
    print("\nSimulating Wealth Inequality Reduction...")
    sim_result = engine.simulate_intervention(
        "United States",
        {"economic_wealth_inequality": -0.2,
         "social_social_mobility": 0.15},
        years=10
    )
    print(f"Projected 10-year risk score: {sim_result['risk_score']:.2f} "
          f"(Change: {sim_result['risk_score'] - analysis['risk_score']:+.2f})")
    
    # Generate visualization
    # Uncomment to display:
    # engine.visualize_risk_trends("United States")
