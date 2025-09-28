"""
Uncertainty Quantification System
Comprehensive uncertainty analysis for psychohistory predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class UncertaintyAnalysis:
    """Comprehensive uncertainty breakdown"""
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Inherent randomness
    measurement_uncertainty: float  # Observer effect
    temporal_uncertainty: float  # Time-dependent uncertainty
    interaction_uncertainty: float  # System complexity
    total_uncertainty: float  # Combined uncertainty
    confidence_level: float  # Overall confidence (1 - total_uncertainty)
    reliability_score: float  # Historical accuracy weighted confidence

class UncertaintyQuantifier:
    """Advanced uncertainty quantification for predictions"""
    
    def __init__(self):
        self.historical_accuracy = {}  # Track accuracy by prediction type
        self.calibration_data = []
        self.baseline_uncertainty = 0.15  # Base uncertainty for all predictions
        
    def quantify_prediction_uncertainty(self, 
                                      prediction_value: float,
                                      prediction_type: str,
                                      input_features: Dict[str, float],
                                      model_ensemble_std: Optional[float] = None,
                                      historical_accuracy: Optional[float] = None) -> UncertaintyAnalysis:
        """
        Comprehensive uncertainty analysis for a prediction
        
        Args:
            prediction_value: The predicted value (0-1)
            prediction_type: Type of prediction (e.g., 'stability', 'risk')
            input_features: Dictionary of input features
            model_ensemble_std: Standard deviation from model ensemble
            historical_accuracy: Historical accuracy for this prediction type
        """
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = self._calculate_epistemic_uncertainty(
            model_ensemble_std, prediction_type)
        
        # Aleatoric uncertainty (inherent randomness)
        aleatoric = self._calculate_aleatoric_uncertainty(
            prediction_value, historical_accuracy or 0.8)
        
        # Measurement uncertainty (observer effect)
        measurement = self._calculate_measurement_uncertainty(
            prediction_value, input_features)
        
        # Temporal uncertainty (time-dependent)
        temporal = self._calculate_temporal_uncertainty(input_features)
        
        # Interaction uncertainty (system complexity)
        interaction = self._calculate_interaction_uncertainty(input_features)
        
        # Combine uncertainties
        total = self._combine_uncertainties(
            epistemic, aleatoric, measurement, temporal, interaction)
        
        # Calculate confidence and reliability
        confidence = max(0.05, 1.0 - total)
        reliability = self._calculate_reliability(
            confidence, prediction_type, historical_accuracy)
        
        return UncertaintyAnalysis(
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            measurement_uncertainty=measurement,
            temporal_uncertainty=temporal,
            interaction_uncertainty=interaction,
            total_uncertainty=total,
            confidence_level=confidence,
            reliability_score=reliability
        )
    
    def _calculate_epistemic_uncertainty(self, 
                                       ensemble_std: Optional[float], 
                                       prediction_type: str) -> float:
        """Calculate model uncertainty"""
        if ensemble_std is not None:
            return min(0.4, float(ensemble_std))
        
        # Default epistemic uncertainty based on prediction type
        type_uncertainties = {
            'stability': 0.12,
            'risk': 0.15,
            'pattern_match': 0.08,
            'timeline': 0.25,
            'intervention_effect': 0.3
        }
        
        return type_uncertainties.get(prediction_type, 0.15)
    
    def _calculate_aleatoric_uncertainty(self, 
                                       prediction_value: float, 
                                       historical_accuracy: float) -> float:
        """Calculate inherent randomness uncertainty"""
        # Higher uncertainty for extreme predictions
        extremeness = abs(prediction_value - 0.5) * 2
        base_aleatoric = 0.1 * (1 - historical_accuracy)
        
        # Uncertainty increases for extreme predictions
        aleatoric = base_aleatoric * (1 + extremeness * 0.5)
        
        return min(0.3, aleatoric)
    
    def _calculate_measurement_uncertainty(self, 
                                         prediction_value: float,
                                         features: Dict[str, float]) -> float:
        """Calculate observer effect uncertainty"""
        
        # Features that affect observability/measurement
        observability_factors = [
            'POLITICAL_institutional_trust',
            'AI_INFLUENCE_information_velocity',
            'AI_INFLUENCE_ai_penetration_rate',
            'TECHNOLOGICAL_information_freedom'
        ]
        
        observability_score = 0.5  # Default
        factor_count = 0
        
        for factor in observability_factors:
            if factor in features:
                observability_score += features[factor]
                factor_count += 1
        
        if factor_count > 0:
            observability_score /= (factor_count + 1)  # +1 for the default 0.5
        
        # Higher observability can increase measurement impact
        # (predictions about highly observed systems can influence outcomes)
        measurement_impact = observability_score * abs(prediction_value - 0.5)
        
        return min(0.25, measurement_impact * 0.5)
    
    def _calculate_temporal_uncertainty(self, features: Dict[str, float]) -> float:
        """Calculate time-dependent uncertainty"""
        
        # Features that affect temporal stability
        volatility_factors = [
            'ECONOMIC_currency_stability',
            'POLITICAL_political_stability', 
            'AI_INFLUENCE_information_velocity',
            'SOCIAL_social_cohesion'
        ]
        
        instability_score = 0.0
        factor_count = 0
        
        for factor in volatility_factors:
            if factor in features:
                if 'stability' in factor or 'cohesion' in factor:
                    # Lower stability = higher instability
                    instability_score += (1.0 - features[factor])
                else:
                    # Higher velocity = higher instability
                    instability_score += features[factor]
                factor_count += 1
        
        if factor_count > 0:
            instability_score /= factor_count
        else:
            instability_score = 0.5  # Default moderate instability
        
        # Base temporal uncertainty increases with instability
        base_temporal = 0.08
        temporal_uncertainty = base_temporal * (1 + instability_score)
        
        return min(0.4, temporal_uncertainty)
    
    def _calculate_interaction_uncertainty(self, features: Dict[str, float]) -> float:
        """Calculate uncertainty from system interactions and complexity"""
        
        # Features that indicate system complexity
        complexity_factors = [
            'AI_INFLUENCE_ai_penetration_rate',
            'AI_INFLUENCE_algorithmic_governance',
            'AI_INFLUENCE_personalized_reality_bubbles',
            'TECHNOLOGICAL_digital_adoption',
            'SOCIAL_urbanization_rate'
        ]
        
        complexity_score = 0.0
        factor_count = 0
        
        for factor in complexity_factors:
            if factor in features:
                complexity_score += features[factor]
                factor_count += 1
        
        if factor_count > 0:
            complexity_score /= factor_count
        else:
            complexity_score = 0.3  # Default low-medium complexity
        
        # Exponential increase in uncertainty with complexity
        interaction_uncertainty = 0.05 * np.exp(complexity_score * 1.5)
        
        return min(0.35, interaction_uncertainty)
    
    def _combine_uncertainties(self, epistemic: float, aleatoric: float, 
                             measurement: float, temporal: float, 
                             interaction: float) -> float:
        """Combine different uncertainty sources"""
        
        # Use root sum of squares for independent uncertainties
        independent_uncertainties = np.sqrt(
            epistemic**2 + aleatoric**2 + temporal**2
        )
        
        # Measurement and interaction uncertainties can be correlated
        dependent_uncertainties = measurement + interaction * 0.7
        
        total = np.sqrt(independent_uncertainties**2 + dependent_uncertainties**2)
        
        # Add baseline uncertainty
        total = np.sqrt(total**2 + self.baseline_uncertainty**2)
        
        return min(0.8, total)  # Cap at 80% uncertainty
    
    def _calculate_reliability(self, confidence: float, 
                             prediction_type: str,
                             historical_accuracy: Optional[float]) -> float:
        """Calculate overall reliability score"""
        
        if historical_accuracy is not None:
            accuracy = historical_accuracy
        else:
            # Default accuracies by prediction type
            type_accuracies = {
                'stability': 0.75,
                'risk': 0.72,
                'pattern_match': 0.68,
                'timeline': 0.60,
                'intervention_effect': 0.55
            }
            accuracy = type_accuracies.get(prediction_type, 0.70)
        
        # Reliability combines accuracy and confidence
        reliability = (accuracy * 0.7 + confidence * 0.3)
        
        return max(0.1, min(0.95, reliability))
    
    def update_historical_accuracy(self, prediction_type: str, 
                                 predicted_value: float,
                                 actual_value: float,
                                 prediction_date: datetime):
        """Update historical accuracy tracking"""
        
        if prediction_type not in self.historical_accuracy:
            self.historical_accuracy[prediction_type] = []
        
        error = abs(predicted_value - actual_value)
        accuracy = max(0.0, 1.0 - error)
        
        self.historical_accuracy[prediction_type].append({
            'accuracy': accuracy,
            'date': prediction_date,
            'predicted': predicted_value,
            'actual': actual_value
        })
        
        # Keep only recent records (last 2 years)
        cutoff_date = datetime.now() - timedelta(days=730)
        self.historical_accuracy[prediction_type] = [
            record for record in self.historical_accuracy[prediction_type]
            if record['date'] > cutoff_date
        ]
    
    def get_historical_accuracy(self, prediction_type: str) -> float:
        """Get historical accuracy for a prediction type"""
        
        if prediction_type not in self.historical_accuracy:
            return 0.7  # Default accuracy
        
        records = self.historical_accuracy[prediction_type]
        if not records:
            return 0.7
        
        # Weight recent records more heavily
        total_weight = 0.0
        weighted_accuracy = 0.0
        
        for record in records:
            days_ago = (datetime.now() - record['date']).days
            weight = np.exp(-days_ago / 365.0)  # Exponential decay over 1 year
            weighted_accuracy += record['accuracy'] * weight
            total_weight += weight
        
        return weighted_accuracy / total_weight if total_weight > 0 else 0.7
    
    def calibrate_confidence(self, predicted_probabilities: List[float],
                           actual_outcomes: List[bool],
                           n_bins: int = 10) -> Dict[str, Any]:
        """Calibrate confidence levels against actual outcomes"""
        
        if len(predicted_probabilities) != len(actual_outcomes):
            raise ValueError("Predicted probabilities and outcomes must have same length")
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        calibration_data = {
            'bin_centers': bin_centers.tolist(),
            'observed_frequencies': [],
            'predicted_frequencies': [],
            'bin_counts': []
        }
        
        for i in range(n_bins):
            # Find predictions in this bin
            in_bin = (predicted_probabilities >= bins[i]) & (predicted_probabilities < bins[i+1])
            
            if i == n_bins - 1:  # Include 1.0 in the last bin
                in_bin = in_bin | (predicted_probabilities == 1.0)
            
            bin_outcomes = [actual_outcomes[j] for j, mask in enumerate(in_bin) if mask]
            bin_predictions = [predicted_probabilities[j] for j, mask in enumerate(in_bin) if mask]
            
            if bin_outcomes:
                observed_freq = sum(bin_outcomes) / len(bin_outcomes)
                predicted_freq = sum(bin_predictions) / len(bin_predictions)
            else:
                observed_freq = 0.0
                predicted_freq = 0.0
            
            calibration_data['observed_frequencies'].append(observed_freq)
            calibration_data['predicted_frequencies'].append(predicted_freq)
            calibration_data['bin_counts'].append(len(bin_outcomes))
        
        # Calculate calibration metrics
        obs_freqs = np.array(calibration_data['observed_frequencies'])
        pred_freqs = np.array(calibration_data['predicted_frequencies'])
        bin_counts = np.array(calibration_data['bin_counts'])
        
        # Reliability (calibration error)
        reliability = np.sum(bin_counts * (obs_freqs - pred_freqs)**2) / sum(bin_counts) if sum(bin_counts) > 0 else 0
        
        # Resolution (ability to discriminate)
        overall_rate = sum(actual_outcomes) / len(actual_outcomes)
        resolution = np.sum(bin_counts * (obs_freqs - overall_rate)**2) / sum(bin_counts) if sum(bin_counts) > 0 else 0
        
        # Brier score
        brier_score = np.mean([(pred - actual)**2 for pred, actual in zip(predicted_probabilities, actual_outcomes)])
        
        calibration_data.update({
            'reliability_score': float(reliability),
            'resolution_score': float(resolution),
            'brier_score': float(brier_score),
            'calibration_quality': 'good' if reliability < 0.02 else 'moderate' if reliability < 0.05 else 'poor',
            'n_samples': len(predicted_probabilities)
        })
        
        return calibration_data
    
    def uncertainty_decomposition(self, uncertainty_analysis: UncertaintyAnalysis) -> Dict[str, float]:
        """Break down uncertainty into interpretable components"""
        total = uncertainty_analysis.total_uncertainty
        
        if total <= 0:
            return {component: 0.0 for component in [
                'model_uncertainty', 'data_uncertainty', 'measurement_effects', 
                'temporal_effects', 'complexity_effects'
            ]}
        
        return {
            'model_uncertainty': uncertainty_analysis.epistemic_uncertainty / total,
            'data_uncertainty': uncertainty_analysis.aleatoric_uncertainty / total,
            'measurement_effects': uncertainty_analysis.measurement_uncertainty / total,
            'temporal_effects': uncertainty_analysis.temporal_uncertainty / total,
            'complexity_effects': uncertainty_analysis.interaction_uncertainty / total
        }
    
    def suggest_uncertainty_reduction(self, uncertainty_analysis: UncertaintyAnalysis) -> List[Dict[str, str]]:
        """Suggest ways to reduce uncertainty"""
        suggestions = []
        
        if uncertainty_analysis.epistemic_uncertainty > 0.15:
            suggestions.append({
                'type': 'model_improvement',
                'suggestion': 'Collect more training data or use ensemble methods',
                'impact': 'high'
            })
        
        if uncertainty_analysis.measurement_uncertainty > 0.1:
            suggestions.append({
                'type': 'measurement_protocol',
                'suggestion': 'Implement secure prediction protocols to minimize observer effects',
                'impact': 'medium'
            })
        
        if uncertainty_analysis.temporal_uncertainty > 0.15:
            suggestions.append({
                'type': 'temporal_modeling',
                'suggestion': 'Incorporate time-series modeling and trend analysis',
                'impact': 'medium'
            })
        
        if uncertainty_analysis.interaction_uncertainty > 0.2:
            suggestions.append({
                'type': 'complexity_reduction',
                'suggestion': 'Focus analysis on key subsystems to reduce interaction complexity',
                'impact': 'low'
            })
        
        return suggestions
    
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get summary statistics about uncertainty quantification"""
        summary = {
            'total_predictions_tracked': sum(len(records) for records in self.historical_accuracy.values()),
            'prediction_types': list(self.historical_accuracy.keys()),
            'average_accuracies': {},
            'baseline_uncertainty': self.baseline_uncertainty
        }
        
        for pred_type in self.historical_accuracy:
            summary['average_accuracies'][pred_type] = self.get_historical_accuracy(pred_type)
        
        return summary
