"""
Psychohistory Metrics System
Comprehensive metrics tracking and management for civilizational analysis
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum, auto
from dataclasses import dataclass, field

class MetricCategory(Enum):
    """Categories of civilizational metrics"""
    ECONOMIC = auto()
    SOCIAL = auto() 
    POLITICAL = auto()
    ENVIRONMENTAL = auto()
    TECHNOLOGICAL = auto()
    AI_INFLUENCE = auto()

@dataclass
class Metric:
    """Individual metric with metadata"""
    value: float = 0.5
    weight: float = 0.2
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    trend: float = 0.0  # Rate of change per year
    volatility: float = 0.0  # Historical volatility
    last_updated: datetime = field(default_factory=datetime.now)

class CivilizationMetrics:
    """Comprehensive metrics tracking system"""
    
    def __init__(self):
        self.metrics = self._initialize_metrics()
        self.historical_data = []
        self.current_date = datetime.now()
    
    def _initialize_metrics(self) -> Dict[MetricCategory, Dict[str, Metric]]:
        """Initialize all metric categories with default values"""
        return {
            MetricCategory.ECONOMIC: {
                'wealth_inequality': Metric(weight=0.25),
                'currency_stability': Metric(weight=0.2),
                'trade_volume': Metric(weight=0.15),
                'debt_to_gdp': Metric(weight=0.25),
                'inflation_rate': Metric(weight=0.15),
                'employment_rate': Metric(weight=0.2),
                'gdp_growth': Metric(weight=0.15),
                'food_security': Metric(weight=0.1)
            },
            MetricCategory.SOCIAL: {
                'civic_engagement': Metric(weight=0.3),
                'social_mobility': Metric(weight=0.25),
                'population_growth': Metric(weight=0.15),
                'urbanization_rate': Metric(weight=0.1),
                'education_index': Metric(weight=0.2),
                'social_cohesion': Metric(weight=0.25),
                'migration_pressure': Metric(weight=0.15),
                'cultural_diversity': Metric(weight=0.1)
            },
            MetricCategory.POLITICAL: {
                'institutional_trust': Metric(weight=0.3),
                'corruption_index': Metric(weight=0.25),
                'political_stability': Metric(weight=0.2),
                'military_spending_ratio': Metric(weight=0.15),
                'democratic_index': Metric(weight=0.1),
                'rule_of_law': Metric(weight=0.2),
                'government_effectiveness': Metric(weight=0.15)
            },
            MetricCategory.ENVIRONMENTAL: {
                'resource_depletion': Metric(weight=0.4),
                'climate_stress': Metric(weight=0.3),
                'agricultural_productivity': Metric(weight=0.2),
                'energy_security': Metric(weight=0.1),
                'biodiversity_loss': Metric(weight=0.15),
                'pollution_index': Metric(weight=0.1),
                'renewable_energy_share': Metric(weight=0.15)
            },
            MetricCategory.TECHNOLOGICAL: {
                'innovation_rate': Metric(weight=0.3),
                'information_freedom': Metric(weight=0.2),
                'digital_adoption': Metric(weight=0.2),
                'scientific_output': Metric(weight=0.3),
                'cybersecurity': Metric(weight=0.15),
                'tech_inequality': Metric(weight=0.1),
                'automation_impact': Metric(weight=0.15)
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
                'collective_intelligence_erosion': Metric(weight=0.3),
                'ai_transparency': Metric(weight=0.1),
                'ai_safety_measures': Metric(weight=0.15)
            }
        }
    
    def update_metric(self, category: MetricCategory, name: str, value: float):
        """Update a specific metric with validation"""
        if not isinstance(category, MetricCategory):
            raise TypeError("category must be a MetricCategory enum")
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(value, (int, float)):
            raise TypeError("value must be a number")
        
        if category not in self.metrics:
            raise ValueError(f"Unknown category: {category}")
        if name not in self.metrics[category]:
            raise ValueError(f"Unknown metric '{name}' in category {category.name}")
        
        metric = self.metrics[category][name]
        
        # Store previous value for trend calculation
        prev_value = metric.value
        
        # Clamp value to valid range
        metric.value = max(metric.min_threshold, 
                          min(metric.max_threshold, float(value)))
        
        # Calculate trend if we have historical data
        if self.historical_data:
            time_diff = (datetime.now() - metric.last_updated).days / 365.25
            if time_diff > 0:
                metric.trend = (metric.value - prev_value) / time_diff
        
        metric.last_updated = datetime.now()
    
    def get_metric(self, category: MetricCategory, name: str) -> float:
        """Get current value of a metric"""
        if category not in self.metrics or name not in self.metrics[category]:
            raise ValueError(f"Unknown metric: {category.name}.{name}")
        return self.metrics[category][name].value
    
    def get_metric_weight(self, category: MetricCategory, name: str) -> float:
        """Get weight of a metric"""
        if category not in self.metrics or name not in self.metrics[category]:
            return 0.0
        return self.metrics[category][name].weight
    
    def get_metrics(self, category: MetricCategory) -> Dict[str, float]:
        """Get all metrics for a category"""
        if category not in self.metrics:
            raise ValueError(f"Unknown category: {category}")
        return {name: metric.value for name, metric in self.metrics[category].items()}
    
    def set_metric(self, category: MetricCategory, name: str, value: float):
        """Alias for update_metric for backward compatibility"""
        self.update_metric(category, name, value)
    
    def take_snapshot(self, snapshot_date: datetime = None) -> Dict:
        """Record current state with timestamp"""
        if snapshot_date is None:
            snapshot_date = datetime.now()
        
        snapshot = {
            'date': snapshot_date,
            'metrics': {
                cat.name: {name: metric.value for name, metric in metrics.items()}
                for cat, metrics in self.metrics.items()
            },
            'trends': {
                cat.name: {name: metric.trend for name, metric in metrics.items()}
                for cat, metrics in self.metrics.items()
            },
            'weights': {
                cat.name: {name: metric.weight for name, metric in metrics.items()}
                for cat, metrics in self.metrics.items()
            }
        }
        
        self.historical_data.append(snapshot)
        self.current_date = snapshot_date
        return snapshot
    
    def calculate_category_score(self, category: MetricCategory) -> float:
        """Calculate weighted average score for a category"""
        if category not in self.metrics:
            raise ValueError(f"Unknown category: {category}")
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric in self.metrics[category].values():
            total_weight += metric.weight
            weighted_sum += metric.value * metric.weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def calculate_overall_score(self) -> float:
        """Calculate overall civilizational score"""
        category_weights = {
            MetricCategory.ECONOMIC: 0.25,
            MetricCategory.SOCIAL: 0.2,
            MetricCategory.POLITICAL: 0.2,
            MetricCategory.ENVIRONMENTAL: 0.15,
            MetricCategory.TECHNOLOGICAL: 0.1,
            MetricCategory.AI_INFLUENCE: 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for category, weight in category_weights.items():
            if category in self.metrics:
                category_score = self.calculate_category_score(category)
                total_score += category_score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def get_critical_metrics(self, threshold: float = 0.7) -> List[Tuple[str, float, str]]:
        """Get metrics that are above/below critical thresholds"""
        critical = []
        
        for category, metrics in self.metrics.items():
            for name, metric in metrics.items():
                if metric.value > threshold or metric.value < (1 - threshold):
                    status = "HIGH" if metric.value > threshold else "LOW"
                    critical.append((f"{category.name}_{name}", metric.value, status))
        
        return sorted(critical, key=lambda x: abs(x[1] - 0.5), reverse=True)
    
    def get_trending_metrics(self, min_trend: float = 0.1) -> List[Tuple[str, float, str]]:
        """Get metrics with significant trends"""
        trending = []
        
        for category, metrics in self.metrics.items():
            for name, metric in metrics.items():
                if abs(metric.trend) > min_trend:
                    direction = "IMPROVING" if metric.trend > 0 else "DETERIORATING"
                    # Flip for negative metrics
                    if name in ['wealth_inequality', 'corruption_index', 'resource_depletion', 
                               'climate_stress', 'debt_to_gdp', 'inflation_rate']:
                        direction = "DETERIORATING" if metric.trend > 0 else "IMPROVING"
                    
                    trending.append((f"{category.name}_{name}", metric.trend, direction))
        
        return sorted(trending, key=lambda x: abs(x[1]), reverse=True)
    
    def export_data(self) -> Dict:
        """Export all metrics data"""
        return {
            'current_metrics': {
                cat.name: {
                    name: {
                        'value': metric.value,
                        'weight': metric.weight,
                        'trend': metric.trend,
                        'last_updated': metric.last_updated.isoformat()
                    }
                    for name, metric in metrics.items()
                }
                for cat, metrics in self.metrics.items()
            },
            'historical_data': [
                {
                    'date': snapshot['date'].isoformat(),
                    'metrics': snapshot['metrics'],
                    'trends': snapshot.get('trends', {}),
                    'weights': snapshot.get('weights', {})
                }
                for snapshot in self.historical_data
            ],
            'scores': {
                'overall': self.calculate_overall_score(),
                'by_category': {
                    cat.name: self.calculate_category_score(cat)
                    for cat in self.metrics.keys()
                }
            }
        }
    
    def import_data(self, data: Dict):
        """Import metrics data"""
        if 'current_metrics' in data:
            for cat_name, metrics_data in data['current_metrics'].items():
                try:
                    category = MetricCategory[cat_name]
                    for metric_name, metric_data in metrics_data.items():
                        if metric_name in self.metrics[category]:
                            metric = self.metrics[category][metric_name]
                            metric.value = metric_data.get('value', metric.value)
                            metric.weight = metric_data.get('weight', metric.weight)
                            metric.trend = metric_data.get('trend', metric.trend)
                            if 'last_updated' in metric_data:
                                metric.last_updated = datetime.fromisoformat(
                                    metric_data['last_updated']
                                )
                except (KeyError, ValueError):
                    continue
        
        if 'historical_data' in data:
            self.historical_data = []
            for snapshot_data in data['historical_data']:
                snapshot = {
                    'date': datetime.fromisoformat(snapshot_data['date']),
                    'metrics': snapshot_data['metrics'],
                    'trends': snapshot_data.get('trends', {}),
                    'weights': snapshot_data.get('weights', {})
                }
                self.historical_data.append(snapshot)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return self.export_data()
    
    def __str__(self) -> str:
        """String representation"""
        overall_score = self.calculate_overall_score()
        critical_metrics = len(self.get_critical_metrics())
        trending_metrics = len(self.get_trending_metrics())
        
        return (f"CivilizationMetrics(overall_score={overall_score:.2f}, "
                f"critical_metrics={critical_metrics}, "
                f"trending_metrics={trending_metrics}, "
                f"snapshots={len(self.historical_data)})")
