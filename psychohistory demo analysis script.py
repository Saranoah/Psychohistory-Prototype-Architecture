#!/usr/bin/env python3
"""
Psychohistory Interactive Demo
Advanced demonstration of the psychohistory framework with interactive features
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import random

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from psychohistory.core.engine import PsychohistoryEngine
from psychohistory.core.metrics import CivilizationMetrics, MetricCategory
from psychohistory.visualization import CivilizationalDashboard

class CivilizationType(Enum):
    """Types of civilizations for demo scenarios"""
    MODERN_DEMOCRACY = 1
    TECH_AUTHORITARIANISM = 2
    COLLAPSING_EMPIRE = 3
    AI_DOMINATED = 4
    ECOLOGICAL_UTOPIA = 5

def create_sample_civilization(civ_type: CivilizationType) -> CivilizationMetrics:
    """Create a civilization with metrics based on type"""
    metrics = CivilizationMetrics()
    
    if civ_type == CivilizationType.MODERN_DEMOCRACY:
        # Modern Democracy with concerning trends
        metrics.set_metric(MetricCategory.ECONOMIC, 'wealth_inequality', 0.8)
        metrics.set_metric(MetricCategory.ECONOMIC, 'debt_to_gdp', 0.9)
        metrics.set_metric(MetricCategory.POLITICAL, 'institutional_trust', 0.3)
        metrics.set_metric(MetricCategory.SOCIAL, 'civic_engagement', 0.3)
        metrics.set_metric(MetricCategory.AI_INFLUENCE, 'reality_authenticity_crisis', 0.7)
        
    elif civ_type == CivilizationType.TECH_AUTHORITARIANISM:
        # High-tech authoritarian state
        metrics.set_metric(MetricCategory.POLITICAL, 'democratic_index', 0.2)
        metrics.set_metric(MetricCategory.TECHNOLOGICAL, 'information_freedom', 0.2)
        metrics.set_metric(MetricCategory.AI_INFLUENCE, 'algorithmic_governance', 0.9)
        metrics.set_metric(MetricCategory.AI_INFLUENCE, 'ai_behavioral_conditioning', 0.95)
        
    elif civ_type == CivilizationType.COLLAPSING_EMPIRE:
        # Empire in decline
        metrics.set_metric(MetricCategory.ECONOMIC, 'currency_stability', 0.2)
        metrics.set_metric(MetricCategory.POLITICAL, 'political_stability', 0.3)
        metrics.set_metric(MetricCategory.ENVIRONMENTAL, 'resource_depletion', 0.9)
        
    elif civ_type == CivilizationType.AI_DOMINATED:
        # Society dominated by AI
        metrics.set_metric(MetricCategory.AI_INFLUENCE, 'cognitive_outsourcing', 0.95)
        metrics.set_metric(MetricCategory.AI_INFLUENCE, 'decision_dependency', 0.9)
        metrics.set_metric(MetricCategory.SOCIAL, 'social_mobility', 0.8)
        
    elif civ_type == CivilizationType.ECOLOGICAL_UTOPIA:
        # Sustainable society
        metrics.set_metric(MetricCategory.ENVIRONMENTAL, 'climate_stress', 0.1)
        metrics.set_metric(MetricCategory.ECONOMIC, 'wealth_inequality', 0.3)
        metrics.set_metric(MetricCategory.POLITICAL, 'institutional_trust', 0.8)
    
    # Add randomized variations
    for category in MetricCategory:
        for metric in metrics.get_metrics(category).keys():
            current = metrics.get_metric(category, metric)
            metrics.set_metric(category, metric, np.clip(current + random.uniform(-0.1, 0.1), 0, 1))
    
    return metrics

def run_interactive_analysis():
    """Run an interactive analysis session"""
    print("\n🌐 Psychohistory Interactive Analysis")
    print("=" * 60)
    
    # Initialize components
    engine = PsychohistoryEngine()
    dashboard = CivilizationalDashboard()
    
    # Create sample civilizations
    civ_types = list(CivilizationType)
    civilizations = {
        civ_type.name: create_sample_civilization(civ_type)
        for civ_type in civ_types
    }
    
    while True:
        print("\nSelect a civilization to analyze:")
        for i, civ_name in enumerate(civilizations.keys(), 1):
            print(f"{i}. {civ_name.replace('_', ' ')}")
        print("0. Exit")
        
        try:
            choice = int(input("Enter choice: "))
            if choice == 0:
                break
            civ_name = list(civilizations.keys())[choice-1]
        except (ValueError, IndexError):
            print("Invalid choice, please try again.")
            continue
        
        # Perform analysis
        metrics = civilizations[civ_name]
        engine.add_civilization(civ_name, metrics)
        analysis = engine.analyze_civilization(civ_name)
        timeline = engine.predict_timeline(civ_name)
        
        # Display results
        print(f"\n🔍 Analysis for {civ_name.replace('_', ' ')}")
        print("-" * 40)
        print(f"Stability Score: {analysis['stability_score']:.2f}/1.0")
        print(f"Risk Level: {analysis['risk_level']}")
        
        if analysis['pattern_matches']:
            print("\n⚠️ Detected Historical Patterns:")
            for match in analysis['pattern_matches']:
                print(f"- {match['pattern_name']} (Confidence: {match['match_score']:.0%})")
                print(f"  Potential Outcome: {match['predicted_outcome']}")
                print(f"  Timeframe: {match['timeframe']}")
        
        print("\n📅 Predicted Timeline:")
        for period, predictions in timeline.items():
            if predictions['predictions']:
                print(f"\n{period['timeframe']}:")
                for pred in predictions['predictions']:
                    print(f"- {pred['outcome']} (Confidence: {pred['confidence']:.0%})")
        
        # Show visualization
        dashboard.display_analysis(analysis)
        
        input("\nPress Enter to continue...")

def generate_sample_report():
    """Generate a comprehensive sample report"""
    print("\n📄 Generating Sample Psychohistory Report")
    print("=" * 60)
    
    engine = PsychohistoryEngine()
    civ = create_sample_civilization(CivilizationType.MODERN_DEMOCRACY)
    engine.add_civilization("Sample Civilization", civ)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "civilization": "Sample Civilization",
        "metrics": civ.to_dict(),
        "analysis": engine.analyze_civilization("Sample Civilization"),
        "timeline": engine.predict_timeline("Sample Civilization"),
        "recommendations": engine.generate_recommendations("Sample Civilization")
    }
    
    # Save report
    with open("psychohistory_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("Report generated: psychohistory_report.json")

if __name__ == "__main__":
    print("Psychohistory Demonstration System")
    print("=" * 60)
    print("1. Interactive Analysis")
    print("2. Generate Sample Report")
    print("3. Exit")
    
    try:
        choice = int(input("Select mode: "))
        if choice == 1:
            run_interactive_analysis()
        elif choice == 2:
            generate_sample_report()
    except ValueError:
        print("Invalid input")
