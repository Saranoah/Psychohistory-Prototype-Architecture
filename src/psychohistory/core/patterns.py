"""
Historical Pattern Recognition System
Pattern matching and management for psychohistory analysis
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

class PatternType(Enum):
    """Types of historical patterns"""
    ECONOMIC_CRISIS = "economic_crisis"
    POLITICAL_UPHEAVAL = "political_upheaval"
    SOCIAL_REVOLUTION = "social_revolution"
    ENVIRONMENTAL_COLLAPSE = "environmental_collapse"
    TECHNOLOGICAL_DISRUPTION = "technological_disruption"
    AI_INTEGRATION = "ai_integration"

@dataclass
class HistoricalExample:
    """An example of a pattern occurring in history"""
    civilization: str
    period: str
    outcome: str
    relevance_score: float = 1.0
    added_date: datetime = field(default_factory=datetime.now)
    data_quality: str = "medium"  # low, medium, high

class HistoricalPattern:
    """Represents a pattern from historical analysis"""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 pattern_type: PatternType,
                 preconditions: Dict[str, Tuple[float, float]],
                 outcome: str,
                 base_confidence: float,
                 timeframe: str,
                 severity: float = 0.7):
        
        self.name = name
        self.description = description
        self.pattern_type = pattern_type
        self.preconditions = preconditions
        self.outcome = outcome
        self.base_confidence = base_confidence
        self.current_confidence = base_confidence
        self.timeframe = timeframe
        self.severity = severity
        self.examples = []
        self.creation_date = datetime.now()
        self.last_updated = datetime.now()
        self.match_history = []
        
    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence required for pattern matching"""
        return self.current_confidence * 0.75
    
    def add_example(self, example: HistoricalExample):
        """Add a historical example and update confidence"""
        self.examples.append(example)
        self._update_confidence()
        self.last_updated = datetime.now()
    
    def _update_confidence(self):
        """Update confidence based on examples and their quality"""
        if not self.examples:
            self.current_confidence = self.base_confidence
            return
        
        # Weight examples by quality and relevance
        quality_weights = {"low": 0.5, "medium": 1.0, "high": 1.5}
        total_weight = 0.0
        weighted_relevance = 0.0
        
        for example in self.examples:
            weight = quality_weights.get(example.data_quality, 1.0)
            total_weight += weight
            weighted_relevance += example.relevance_score * weight
        
        if total_weight > 0:
            avg_relevance = weighted_relevance / total_weight
            # Boost confidence with more high-quality examples
            confidence_boost = min(0.2, len(self.examples) * 0.03 * avg_relevance)
            self.current_confidence = min(0.95, self.base_confidence + confidence_boost)
        
    def get_metric_weight(self, metric_key: str) -> float:
        """Get weight for a specific metric in pattern matching"""
        # Equal weights for now - could be enhanced with learned weights
        return 1.0 if metric_key in self.preconditions else 0.0
    
    def record_match(self, civilization: str, match_score: float, outcome_occurred: bool = None):
        """Record when this pattern was matched"""
        self.match_history.append({
            'civilization': civilization,
            'match_score': match_score,
            'date': datetime.now(),
            'outcome_occurred': outcome_occurred
        })
        
        # Adjust confidence based on outcome accuracy
        if outcome_occurred is not None:
            if outcome_occurred and match_score > 0.7:
                # Pattern correctly predicted outcome
                self.current_confidence = min(0.95, self.current_confidence + 0.02)
            elif not outcome_occurred and match_score > 0.7:
                # Pattern incorrectly predicted outcome
                self.current_confidence = max(0.1, self.current_confidence - 0.05)
    
    def get_accuracy_score(self) -> float:
        """Calculate historical accuracy of this pattern"""
        if not self.match_history:
            return 0.5  # Unknown accuracy
        
        confirmed_matches = [m for m in self.match_history if m['outcome_occurred'] is not None]
        if not confirmed_matches:
            return 0.5
        
        correct_predictions = sum(1 for m in confirmed_matches if m['outcome_occurred'])
        return correct_predictions / len(confirmed_matches)
    
    def to_dict(self) -> Dict:
        """Convert pattern to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'pattern_type': self.pattern_type.value,
            'preconditions': self.preconditions,
            'outcome': self.outcome,
            'base_confidence': self.base_confidence,
            'current_confidence': self.current_confidence,
            'timeframe': self.timeframe,
            'severity': self.severity,
            'examples': [
                {
                    'civilization': ex.civilization,
                    'period': ex.period,
                    'outcome': ex.outcome,
                    'relevance_score': ex.relevance_score,
                    'data_quality': ex.data_quality
                }
                for ex in self.examples
            ],
            'accuracy_score': self.get_accuracy_score(),
            'match_count': len(self.match_history)
        }

class PatternManager:
    """Manages collection of historical patterns"""
    
    def __init__(self):
        self.patterns = []
        self._initialize_core_patterns()
    
    def _initialize_core_patterns(self):
        """Initialize with core historical patterns"""
        
        # Economic Crisis Pattern
        economic_crisis = HistoricalPattern(
            name="Economic Collapse Cycle",
            description="Conditions preceding major economic system failure",
            pattern_type=PatternType.ECONOMIC_CRISIS,
            preconditions={
                'ECONOMIC_wealth_inequality': (0.75, 1.0),
                'ECONOMIC_debt_to_gdp': (0.85, 1.0),
                'ECONOMIC_currency_stability': (0.0, 0.4),
                'POLITICAL_institutional_trust': (0.0, 0.5)
            },
            outcome="Currency crisis, banking collapse, and severe economic recession",
            base_confidence=0.78,
            timeframe="2-8 years",
            severity=0.85
        )
        
        # Add historical examples
        economic_crisis.add_example(HistoricalExample(
            "Weimar Germany", "1920-1923", "Hyperinflation and economic collapse",
            relevance_score=0.9, data_quality="high"
        ))
        economic_crisis.add_example(HistoricalExample(
            "Argentina", "2001", "Currency crisis and default",
            relevance_score=0.85, data_quality="high"
        ))
        economic_crisis.add_example(HistoricalExample(
            "Global Financial Crisis", "2008", "Banking system collapse",
            relevance_score=0.95, data_quality="high"
        ))
        
        # Revolutionary Conditions Pattern
        revolution = HistoricalPattern(
            name="Revolutionary Conditions",
            description="Social and political precursors to revolution or uprising",
            pattern_type=PatternType.SOCIAL_REVOLUTION,
            preconditions={
                'ECONOMIC_wealth_inequality': (0.8, 1.0),
                'SOCIAL_social_mobility': (0.0, 0.3),
                'POLITICAL_institutional_trust': (0.0, 0.4),
                'SOCIAL_civic_engagement': (0.6, 1.0)  # High engagement in crisis
            },
            outcome="Mass uprising, civil war, or regime change",
            base_confidence=0.72,
            timeframe="1-5 years",
            severity=0.9
        )
        
        revolution.add_example(HistoricalExample(
            "French Revolution", "1789-1799", "Overthrow of monarchy",
            relevance_score=0.8, data_quality="high"
        ))
        revolution.add_example(HistoricalExample(
            "Arab Spring", "2010-2012", "Multiple regime changes",
            relevance_score=0.9, data_quality="high"
        ))
        
        # Imperial Decline Pattern
        imperial_decline = HistoricalPattern(
            name="Imperial Overstretch",
            description="Signs of hegemonic power decline",
            pattern_type=PatternType.POLITICAL_UPHEAVAL,
            preconditions={
                'POLITICAL_military_spending_ratio': (0.7, 1.0),
                'ECONOMIC_debt_to_gdp': (0.7, 1.0),
                'POLITICAL_political_stability': (0.0, 0.5),
                'ECONOMIC_trade_volume': (0.0, 0.4)  # Declining trade
            },
            outcome="Gradual loss of global influence and power",
            base_confidence=0.8,
            timeframe="10-50 years",
            severity=0.75
        )
        
        imperial_decline.add_example(HistoricalExample(
            "British Empire", "1914-1956", "Decline from global hegemon",
            relevance_score=0.85, data_quality="high"
        ))
        imperial_decline.add_example(HistoricalExample(
            "Soviet Union", "1970-1991", "Economic stagnation and collapse",
            relevance_score=0.9, data_quality="high"
        ))
        
        # AI Integration Disruption Pattern
        ai_disruption = HistoricalPattern(
            name="AI Cognitive Disruption",
            description="Societal instability from rapid AI integration",
            pattern_type=PatternType.AI_INTEGRATION,
            preconditions={
                'AI_INFLUENCE_cognitive_outsourcing': (0.7, 1.0),
                'AI_INFLUENCE_reality_authenticity_crisis': (0.7, 1.0),
                'AI_INFLUENCE_collective_intelligence_erosion': (0.6, 1.0),
                'SOCIAL_social_cohesion': (0.0, 0.4)
            },
            outcome="Cognitive disorientation and social fragmentation",
            base_confidence=0.65,  # Lower due to limited historical data
            timeframe="5-15 years",
            severity=0.8
        )
        
        ai_disruption.add_example(HistoricalExample(
            "Social Media Era", "2010-2020", "Increased polarization and misinformation",
            relevance_score=0.7, data_quality="medium"
        ))
        
        # Environmental Collapse Pattern
        env_collapse = HistoricalPattern(
            name="Environmental System Collapse",
            description="Ecological degradation leading to civilizational stress",
            pattern_type=PatternType.ENVIRONMENTAL_COLLAPSE,
            preconditions={
                'ENVIRONMENTAL_climate_stress': (0.8, 1.0),
                'ENVIRONMENTAL_resource_depletion': (0.7, 1.0),
                'ENVIRONMENTAL_agricultural_productivity': (0.0, 0.3),
                'SOCIAL_migration_pressure': (0.6, 1.0)
            },
            outcome="Population displacement, resource conflicts, societal breakdown",
            base_confidence=0.7,
            timeframe="10-30 years",
            severity=0.9
        )
        
        env_collapse.add_example(HistoricalExample(
            "Maya Civilization", "800-900 CE", "Drought and agricultural collapse",
            relevance_score=0.75, data_quality="medium"
        ))
        env_collapse.add_example(HistoricalExample(
            "Easter Island", "1600-1722", "Deforestation and population collapse",
            relevance_score=0.7, data_quality="medium"
        ))
        
        # Add all patterns
        self.patterns.extend([
            economic_crisis, revolution, imperial_decline, 
            ai_disruption, env_collapse
        ])
    
    def add_pattern(self, pattern: HistoricalPattern):
        """Add a new pattern to the collection"""
        if not isinstance(pattern, HistoricalPattern):
            raise TypeError("pattern must be a HistoricalPattern instance")
        self.patterns.append(pattern)
    
    def get_pattern(self, name: str) -> Optional[HistoricalPattern]:
        """Get a pattern by name"""
        for pattern in self.patterns:
            if pattern.name == name:
                return pattern
        return None
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[HistoricalPattern]:
        """Get all patterns of a specific type"""
        return [p for p in self.patterns if p.pattern_type == pattern_type]
    
    def get_active_patterns(self, min_confidence: float = 0.5) -> List[HistoricalPattern]:
        """Get patterns above minimum confidence threshold"""
        return [p for p in self.patterns if p.current_confidence >= min_confidence]
    
    def get_patterns_by_severity(self, min_severity: float = 0.7) -> List[HistoricalPattern]:
        """Get patterns above minimum severity threshold"""
        return [p for p in self.patterns if p.severity >= min_severity]
    
    def update_pattern_from_outcome(self, pattern_name: str, civilization: str, 
                                  outcome_occurred: bool, match_score: float):
        """Update pattern confidence based on actual outcome"""
        pattern = self.get_pattern(pattern_name)
        if pattern:
            pattern.record_match(civilization, match_score, outcome_occurred)
    
    def discover_new_pattern(self, name: str, preconditions: Dict[str, Tuple[float, float]], 
                           outcome: str, examples: List[HistoricalExample]) -> HistoricalPattern:
        """Create a new pattern from discovered conditions"""
        pattern = HistoricalPattern(
            name=name,
            description=f"Pattern discovered from data: {name}",
            pattern_type=PatternType.ECONOMIC_CRISIS,  # Default, should be determined
            preconditions=preconditions,
            outcome=outcome,
            base_confidence=0.6,  # Conservative starting confidence
            timeframe="5-15 years",  # Default timeframe
            severity=0.7
        )
        
        for example in examples:
            pattern.add_example(example)
        
        self.add_pattern(pattern)
        return pattern
    
    def get_pattern_statistics(self) -> Dict:
        """Get statistics about the pattern collection"""
        if not self.patterns:
            return {'total_patterns': 0}
        
        by_type = {}
        for pattern_type in PatternType:
            by_type[pattern_type.value] = len(self.get_patterns_by_type(pattern_type))
        
        confidences = [p.current_confidence for p in self.patterns]
        severities = [p.severity for p in self.patterns]
        accuracies = [p.get_accuracy_score() for p in self.patterns]
        
        return {
            'total_patterns': len(self.patterns),
            'by_type': by_type,
            'confidence_stats': {
                'mean': np.mean(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'severity_stats': {
                'mean': np.mean(severities),
                'min': np.min(severities),
                'max': np.max(severities)
            },
            'accuracy_stats': {
                'mean': np.mean(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies)
            },
            'active_patterns': len(self.get_active_patterns()),
            'high_severity_patterns': len(self.get_patterns_by_severity())
        }
    
    def export_patterns(self) -> List[Dict]:
        """Export all patterns to dictionary format"""
        return [pattern.to_dict() for pattern in self.patterns]
    
    def import_patterns(self, patterns_data: List[Dict]):
        """Import patterns from dictionary format"""
        for data in patterns_data:
            try:
                pattern = HistoricalPattern(
                    name=data['name'],
                    description=data['description'],
                    pattern_type=PatternType(data['pattern_type']),
                    preconditions=data['preconditions'],
                    outcome=data['outcome'],
                    base_confidence=data['base_confidence'],
                    timeframe=data['timeframe'],
                    severity=data['severity']
                )
                
                # Add examples
                for ex_data in data.get('examples', []):
                    example = HistoricalExample(
                        civilization=ex_data['civilization'],
                        period=ex_data['period'],
                        outcome=ex_data['outcome'],
                        relevance_score=ex_data.get('relevance_score', 1.0),
                        data_quality=ex_data.get('data_quality', 'medium')
                    )
                    pattern.add_example(example)
                
                self.add_pattern(pattern)
                
            except (KeyError, ValueError) as e:
                print(f"Error importing pattern: {e}")
                continue
