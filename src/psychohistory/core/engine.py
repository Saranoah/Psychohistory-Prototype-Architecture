"""
Psychohistory Core Engine
Main analysis engine with pattern matching and prediction capabilities
"""

import numpy as np
import pandas as pd 
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass
import warnings

from .metrics import CivilizationMetrics, MetricCategory
from .patterns import HistoricalPattern, PatternManager
from .uncertainty import UncertaintyQuantifier

warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class AnalysisResult:
    """Results from civilizational analysis"""
    date: datetime
    risk_score: float
    risk_level: str
    pattern_matches: List[Dict]
    recommendations: List[Dict]
    metric_snapshot: Dict[str, float]
    metric_trends: Dict[str, float]
    uncertainty_analysis: Optional[Dict] = None

class PsychohistoryEngine:
    """Core analysis engine for civilizational monitoring"""
    
    def __init__(self):
        self.pattern_manager = PatternManager()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.civilizations = {}
        self.temporal_resolution = timedelta(days=90)
        self.risk_thresholds = {
            'LOW': 0.4,
            'MEDIUM': 0.7, 
            'HIGH': 0.9
        }
        
    def add_civilization(self, name: str, metrics: CivilizationMetrics):
        """Register a civilization for analysis"""
        if not isinstance(metrics, CivilizationMetrics):
            raise TypeError("metrics must be a CivilizationMetrics instance")
            
        self.civilizations[name] = {
            'metrics': metrics,
            'analyses': [],
            'risk_history': [],
            'intervention_history': []
        }
    
    def analyze_civilization(self, civ_name: str, analysis_date: datetime = None) -> AnalysisResult:
        """Perform comprehensive analysis of a civilization"""
        if civ_name not in self.civilizations:
            raise ValueError(f"Unknown civilization: {civ_name}")
            
        if analysis_date is None:
            analysis_date = datetime.now()
            
        civ = self.civilizations[civ_name]
        metrics = civ['metrics']
        
        # Ensure we have a current snapshot
        if (not metrics.historical_data or 
            (analysis_date - metrics.historical_data[-1]['date']) >= self.temporal_resolution):
            metrics.take_snapshot(analysis_date)
        
        current_state = metrics.historical_data[-1]['metrics']
        current_trends = metrics.historical_data[-1].get('trends', {})
        
        # Normalize metrics
        normalized = self._normalize_metrics(current_state, current_trends)
        
        # Pattern matching
        pattern_matches = self._match_patterns(normalized, civ_name)
        
        # Risk assessment
        risk_score = self._calculate_risk_score(normalized, pattern_matches)
        risk_level = self._determine_risk_level(risk_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(normalized, pattern_matches)
        
        # Create analysis result
        result = AnalysisResult(
            date=analysis_date,
            risk_score=risk_score,
            risk_level=risk_level,
            pattern_matches=pattern_matches,
            recommendations=recommendations,
            metric_snapshot=normalized['values'],
            metric_trends=normalized['trends']
        )
        
        # Store analysis
        civ['analyses'].append(result)
        civ['risk_history'].append((analysis_date, risk_score))
        
        return result
    
    def _normalize_metrics(self, values: Dict, trends: Dict) -> Dict:
        """Normalize metrics with trend analysis"""
        normalized = {
            'values': {},
            'trends': {},
            'projected': {}
        }
        
        # Flatten nested structure
        for category, metrics in values.items():
            for metric, value in metrics.items():
                key = f"{category}_{metric}"
                trend = trends.get(category, {}).get(metric, 0.0)
                
                # Ensure valid values
                value = max(0.0, min(1.0, float(value)))
                trend = float(trend)
                
                normalized['values'][key] = value
                normalized['trends'][key] = trend
                normalized['projected'][key] = max(0.0, min(1.0, value + trend))
        
        return normalized
    
    def _match_patterns(self, metrics: Dict, civ_name: str) -> List[Dict]:
        """Match current state against historical patterns"""
        matches = []
        
        for pattern in self.pattern_manager.get_active_patterns():
            match_score = self._calculate_pattern_match(metrics, pattern)
            
            if match_score >= pattern.confidence_threshold:
                matches.append({
                    'pattern_name': pattern.name,
                    'match_score': match_score,
                    'predicted_outcome': pattern.outcome,
                    'timeframe': pattern.timeframe,
                    'severity': pattern.severity,
                    'confidence': pattern.current_confidence
                })
        
        # Sort by severity and match score
        matches.sort(key=lambda x: (x['severity'], x['match_score']), reverse=True)
        return matches
    
    def _calculate_pattern_match(self, metrics: Dict, pattern: HistoricalPattern) -> float:
        """Calculate how well metrics match a historical pattern"""
        total_weight = 0.0
        matched_weight = 0.0
        
        for metric_key, (min_val, max_val) in pattern.preconditions.items():
            if metric_key in metrics['values']:
                weight = pattern.get_metric_weight(metric_key)
                total_weight += weight
                
                current_value = metrics['values'][metric_key]
                projected_value = metrics['projected'][metric_key]
                
                # Use weighted average of current and projected
                effective_value = current_value * 0.7 + projected_value * 0.3
                
                if min_val <= effective_value <= max_val:
                    matched_weight += weight
                else:
                    # Partial match based on distance
                    if effective_value < min_val:
                        distance = min_val - effective_value
                    else:
                        distance = effective_value - max_val
                    
                    # Exponential decay for partial matches
                    partial_match = np.exp(-distance * 5)
                    matched_weight += weight * partial_match
        
        return matched_weight / total_weight if total_weight > 0 else 0.0
    
    def _calculate_risk_score(self, metrics: Dict, pattern_matches: List[Dict]) -> float:
        """Calculate composite risk score"""
        # Base risk from critical metrics
        critical_metrics = [
            'ECONOMIC_wealth_inequality',
            'POLITICAL_institutional_trust', 
            'SOCIAL_civic_engagement',
            'ENVIRONMENTAL_climate_stress'
        ]
        
        metric_risks = []
        for metric in critical_metrics:
            if metric in metrics['values']:
                value = metrics['values'][metric]
                # Convert to risk (higher values = higher risk for negative metrics)
                if 'trust' in metric or 'engagement' in metric:
                    risk = 1.0 - value  # Lower trust/engagement = higher risk
                else:
                    risk = value  # Higher inequality/stress = higher risk
                metric_risks.append(risk)
        
        base_risk = np.mean(metric_risks) if metric_risks else 0.5
        
        # Pattern-based risk adjustment
        pattern_risk = 0.0
        if pattern_matches:
            # Weight by severity and confidence
            pattern_risk = max([
                match['severity'] * match['match_score'] * match['confidence']
                for match in pattern_matches
            ])
        
        # Combine with trends
        trend_amplifier = 1.0
        negative_trends = [
            t for t in metrics['trends'].values() 
            if t > 0.1  # Worsening trends
        ]
        if negative_trends:
            trend_amplifier = 1.0 + 0.2 * np.mean(negative_trends)
        
        composite_risk = (base_risk * 0.6 + pattern_risk * 0.4) * trend_amplifier
        return min(1.0, composite_risk)
    
    def _determine_risk_level(self, score: float) -> str:
        """Convert risk score to categorical level"""
        if score >= 0.95:
            return "CRITICAL"
        elif score >= self.risk_thresholds['HIGH']:
            return "HIGH"
        elif score >= self.risk_thresholds['MEDIUM']:
            return "MEDIUM"
        elif score >= self.risk_thresholds['LOW']:
            return "LOW"
        else:
            return "STABLE"
    
    def _generate_recommendations(self, metrics: Dict, pattern_matches: List[Dict]) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Metric-based recommendations
        metric_recs = {
            'ECONOMIC_wealth_inequality': {
                'threshold': 0.7,
                'actions': [
                    ("Implement progressive taxation", 0.7, "fiscal_policy"),
                    ("Strengthen social safety nets", 0.6, "social_policy")
                ]
            },
            'POLITICAL_institutional_trust': {
                'threshold': 0.4,
                'actions': [
                    ("Increase government transparency", 0.8, "governance"),
                    ("Combat corruption", 0.7, "governance")
                ]
            },
            'SOCIAL_civic_engagement': {
                'threshold': 0.4,
                'actions': [
                    ("Promote civic education", 0.6, "education"),
                    ("Create citizen participation forums", 0.5, "governance")
                ]
            }
        }
        
        for metric, config in metric_recs.items():
            if metric in metrics['values'] and metrics['values'][metric] > config['threshold']:
                for action, efficacy, category in config['actions']:
                    recommendations.append({
                        'action': action,
                        'category': category,
                        'efficacy': efficacy,
                        'urgency': 'medium',
                        'target_metric': metric
                    })
        
        # Pattern-specific recommendations
        pattern_recs = {
            'Economic Collapse Cycle': [
                ("Establish currency stabilization fund", 0.8, "urgent"),
                ("Implement debt restructuring", 0.7, "medium")
            ],
            'Revolutionary Conditions': [
                ("Address inequality through redistribution", 0.7, "urgent"),
                ("Strengthen democratic institutions", 0.6, "medium")
            ]
        }
        
        for match in pattern_matches:
            pattern_name = match['pattern_name']
            if pattern_name in pattern_recs:
                for action, efficacy, urgency in pattern_recs[pattern_name]:
                    recommendations.append({
                        'action': action,
                        'category': 'pattern_response',
                        'efficacy': efficacy * match['confidence'],
                        'urgency': urgency,
                        'target_pattern': pattern_name
                    })
        
        # Remove duplicates and sort by efficacy
        unique_recs = {}
        for rec in recommendations:
            key = rec['action']
            if key not in unique_recs or rec['efficacy'] > unique_recs[key]['efficacy']:
                unique_recs[key] = rec
        
        return sorted(unique_recs.values(), key=lambda x: x['efficacy'], reverse=True)
    
    def predict_timeline(self, civ_name: str, time_horizon: int = 5) -> List[Dict]:
        """Generate timeline predictions"""
        if civ_name not in self.civilizations:
            raise ValueError(f"Unknown civilization: {civ_name}")
        
        analysis = self.analyze_civilization(civ_name)
        timeline = []
        
        for year in range(1, time_horizon + 1):
            # Simple timeline generation based on current risk
            risk_progression = analysis.risk_score * (1 + 0.1 * year)
            risk_progression = min(1.0, risk_progression)
            
            predictions = []
            
            # Base predictions on pattern matches
            for match in analysis.pattern_matches[:3]:  # Top 3 patterns
                outcome_prob = match['match_score'] * (1 - 0.1 * year)
                predictions.append({
                    'outcome': match['predicted_outcome'],
                    'confidence': max(0.1, outcome_prob)
                })
            
            # Add default prediction if no patterns
            if not predictions:
                if risk_progression > 0.7:
                    predictions.append({
                        'outcome': 'Continued deterioration of stability',
                        'confidence': 0.6
                    })
                else:
                    predictions.append({
                        'outcome': 'Gradual improvement or stability',
                        'confidence': 0.5
                    })
            
            timeline.append({
                'timeframe': f"{year} year{'s' if year > 1 else ''}",
                'predictions': predictions,
                'risk_level': self._determine_risk_level(risk_progression)
            })
        
        return timeline
    
    def generate_recommendations(self, civ_name: str) -> List[Dict]:
        """Generate recommendations for a civilization"""
        analysis = self.analyze_civilization(civ_name)
        return analysis.recommendations
    
    def simulate_intervention(self, civ_name: str, intervention: Dict[str, float], 
                            years: int = 5) -> Dict:
        """Simulate impact of policy interventions"""
        if civ_name not in self.civilizations:
            raise ValueError(f"Unknown civilization: {civ_name}")
        
        # Get current analysis
        baseline = self.analyze_civilization(civ_name)
        
        # Apply intervention to metrics (simplified)
        civ = self.civilizations[civ_name]
        modified_metrics = CivilizationMetrics()
        
        # Copy current state and apply interventions
        current_snapshot = civ['metrics'].historical_data[-1]
        for category, metrics in current_snapshot['metrics'].items():
            for metric, value in metrics.items():
                intervention_key = f"{category}_{metric}"
                effect = intervention.get(intervention_key, 0.0)
                
                # Apply diminishing returns
                adjusted_effect = effect * (1 - abs(effect) * 0.3)
                new_value = max(0.0, min(1.0, value + adjusted_effect))
                
                # Update metric
                try:
                    category_enum = MetricCategory[category]
                    modified_metrics.update_metric(category_enum, metric, new_value)
                except (KeyError, AttributeError):
                    continue
        
        # Create temporary engine for projection
        temp_engine = PsychohistoryEngine()
        temp_engine.add_civilization("projection", modified_metrics)
        projected = temp_engine.analyze_civilization("projection", 
                                                   datetime.now() + timedelta(days=365 * years))
        
        return {
            'baseline_risk': baseline.risk_score,
            'projected_risk': projected.risk_score,
            'risk_change': projected.risk_score - baseline.risk_score,
            'intervention_effectiveness': abs(projected.risk_score - baseline.risk_score),
            'projected_patterns': projected.pattern_matches,
            'time_horizon': years
        }
def main():
    print("Psychohistory Core Engine is running!")
