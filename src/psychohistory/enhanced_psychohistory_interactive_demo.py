#!/usr/bin/env python3
"""
Enhanced Psychohistory Interactive Demo
Now with timeline visualization, shock simulation, and comparative analysis
"""

import sys
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from enum import Enum
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import core components
from psychohistory.core.engine import PsychohistoryEngine
from psychohistory.core.metrics import CivilizationMetrics, MetricCategory
from psychohistory.visualization import CivilizationalDashboard

class CivilizationType(Enum):
    """Expanded civilization types including modern crisis scenarios"""
    MODERN_DEMOCRACY = 1
    TECH_AUTHORITARIANISM = 2
    COLLAPSING_EMPIRE = 3
    AI_DOMINATED = 4
    ECOLOGICAL_UTOPIA = 5
    CLIMATE_COLLAPSE = 6
    POST_PANDEMIC = 7
    CYBERWAR_STATE = 8

def create_sample_civilization(civ_type: CivilizationType) -> CivilizationMetrics:
    """Create a civilization with metrics based on type"""
    metrics = CivilizationMetrics()
    
    # Base configurations
    if civ_type == CivilizationType.MODERN_DEMOCRACY:
        metrics.set_metric(MetricCategory.ECONOMIC, 'wealth_inequality', 0.8)
        metrics.set_metric(MetricCategory.ECONOMIC, 'debt_to_gdp', 0.9)
        metrics.set_metric(MetricCategory.POLITICAL, 'institutional_trust', 0.3)
        metrics.set_metric(MetricCategory.SOCIAL, 'civic_engagement', 0.3)
        metrics.set_metric(MetricCategory.AI_INFLUENCE, 'reality_authenticity_crisis', 0.7)
        
    elif civ_type == CivilizationType.TECH_AUTHORITARIANISM:
        metrics.set_metric(MetricCategory.POLITICAL, 'democratic_index', 0.2)
        metrics.set_metric(MetricCategory.TECHNOLOGICAL, 'information_freedom', 0.2)
        metrics.set_metric(MetricCategory.AI_INFLUENCE, 'algorithmic_governance', 0.9)
        metrics.set_metric(MetricCategory.AI_INFLUENCE, 'ai_behavioral_conditioning', 0.95)
        
    elif civ_type == CivilizationType.COLLAPSING_EMPIRE:
        metrics.set_metric(MetricCategory.ECONOMIC, 'currency_stability', 0.2)
        metrics.set_metric(MetricCategory.POLITICAL, 'political_stability', 0.3)
        metrics.set_metric(MetricCategory.ENVIRONMENTAL, 'resource_depletion', 0.9)
        
    elif civ_type == CivilizationType.AI_DOMINATED:
        metrics.set_metric(MetricCategory.AI_INFLUENCE, 'cognitive_outsourcing', 0.95)
        metrics.set_metric(MetricCategory.AI_INFLUENCE, 'decision_dependency', 0.9)
        metrics.set_metric(MetricCategory.SOCIAL, 'social_mobility', 0.8)
        
    elif civ_type == CivilizationType.ECOLOGICAL_UTOPIA:
        metrics.set_metric(MetricCategory.ENVIRONMENTAL, 'climate_stress', 0.1)
        metrics.set_metric(MetricCategory.ECONOMIC, 'wealth_inequality', 0.3)
        metrics.set_metric(MetricCategory.POLITICAL, 'institutional_trust', 0.8)
    
    # New crisis scenarios
    elif civ_type == CivilizationType.CLIMATE_COLLAPSE:
        metrics.set_metric(MetricCategory.ENVIRONMENTAL, 'climate_stress', 0.95)
        metrics.set_metric(MetricCategory.ENVIRONMENTAL, 'biodiversity_loss', 0.9)
        metrics.set_metric(MetricCategory.SOCIAL, 'migration_pressure', 0.85)
        metrics.set_metric(MetricCategory.ECONOMIC, 'food_security', 0.2)
        
    elif civ_type == CivilizationType.POST_PANDEMIC:
        metrics.set_metric(MetricCategory.HEALTH, 'healthcare_capacity', 0.3)
        metrics.set_metric(MetricCategory.ECONOMIC, 'supply_chain_resilience', 0.4)
        metrics.set_metric(MetricCategory.SOCIAL, 'social_cohesion', 0.5)
        metrics.set_metric(MetricCategory.TECHNOLOGICAL, 'remote_work_infrastructure', 0.8)
        
    elif civ_type == CivilizationType.CYBERWAR_STATE:
        metrics.set_metric(MetricCategory.TECHNOLOGICAL, 'cybersecurity', 0.2)
        metrics.set_metric(MetricCategory.POLITICAL, 'institutional_trust', 0.3)
        metrics.set_metric(MetricCategory.ECONOMIC, 'digital_economy_share', 0.7)
        metrics.set_metric(MetricCategory.AI_INFLUENCE, 'ai_cyber_warfare', 0.85)
    
    # Add randomized variations
    for category in MetricCategory:
        for metric in metrics.get_metrics(category).keys():
            current = metrics.get_metric(category, metric)
            metrics.set_metric(category, metric, np.clip(current + random.uniform(-0.1, 0.1), 0, 1))
    
    return metrics

def display_timeline(timeline: Dict, civ_name: str):
    """Visualize predicted timeline with enhanced styling"""
    plt.figure(figsize=(12, 6), facecolor='#0f1a26')
    ax = plt.gca()
    ax.set_facecolor('#0f1a26')
    
    # Prepare data
    timepoints = []
    confidences = []
    colors = []
    labels = []
    
    # Color mapping for prediction types
    color_map = {
        'positive': '#2ca02c',  # Green
        'negative': '#d62728',   # Red
        'neutral': '#7f7f7f',    # Gray
        'critical': '#ff7f0e'    # Orange
    }
    
    for period in timeline:
        for prediction in period['predictions']:
            timepoints.append(period['timeframe'])
            confidences.append(prediction['confidence'])
            
            # Determine color based on prediction type
            if 'collapse' in prediction['outcome'].lower():
                color = color_map['critical']
            elif any(word in prediction['outcome'].lower() for word in ['growth', 'improve', 'positive']):
                color = color_map['positive']
            elif any(word in prediction['outcome'].lower() for word in ['decline', 'crisis', 'negative']):
                color = color_map['negative']
            else:
                color = color_map['neutral']
                
            colors.append(color)
            labels.append(prediction['outcome'])
    
    # Create bars
    bars = ax.bar(timepoints, confidences, color=colors, edgecolor='#1a2a3a', linewidth=1.2)
    
    # Add text labels
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f"{label[:30]}..." if len(label) > 30 else label,
                ha='center', va='bottom', rotation=45, fontsize=8, color='#e0e0e0')
    
    # Styling
    ax.set_title(f'Civilization Timeline: {civ_name.replace("_", " ")}', 
                color='#e0e0e0', fontsize=14, pad=20)
    ax.set_ylabel('Prediction Confidence', color='#e0e0e0')
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis='x', labelrotation=45, colors='#e0e0e0')
    ax.tick_params(axis='y', colors='#e0e0e0')
    
    # Grid and spine styling
    ax.grid(True, linestyle='--', alpha=0.2, color='#3498db')
    for spine in ax.spines.values():
        spine.set_color('#3498db')
        spine.set_linewidth(0.5)
    
    # Add legend
    legend_elements = [
        Rectangle((0,0), 1, 1, color=color_map['positive'], label='Positive'),
        Rectangle((0,0), 1, 1, color=color_map['negative'], label='Negative'),
        Rectangle((0,0), 1, 1, color=color_map['neutral'], label='Neutral'),
        Rectangle((0,0), 1, 1, color=color_map['critical'], label='Critical')
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='#0f1a26', 
             edgecolor='#3498db', labelcolor='#e0e0e0')
    
    plt.tight_layout()
    plt.show()

def simulate_shock_event(metrics: CivilizationMetrics, event_type: str):
    """Apply random shock to civilization with realistic effects"""
    shock_effects = {
        'FINANCIAL_CRASH': {
            'category': MetricCategory.ECONOMIC,
            'metrics': {
                'currency_stability': -0.4,
                'debt_to_gdp': +0.3,
                'employment_rate': -0.35
            },
            'description': 'Major stock market collapse and banking crisis'
        },
        'AI_REVOLT': {
            'category': MetricCategory.AI_INFLUENCE,
            'metrics': {
                'decision_dependency': -0.5,
                'algorithmic_governance': -0.6,
                'system_trust': -0.45
            },
            'description': 'AI systems develop unexpected goals and resist control'
        },
        'CLIMATE_DISASTER': {
            'category': MetricCategory.ENVIRONMENTAL,
            'metrics': {
                'climate_stress': +0.5,
                'food_security': -0.4,
                'migration_pressure': +0.6
            },
            'description': 'Unprecedented climate event causes widespread destruction'
        },
        'PANDEMIC': {
            'category': MetricCategory.HEALTH,
            'metrics': {
                'healthcare_capacity': -0.7,
                'social_cohesion': -0.5,
                'supply_chain_resilience': -0.6
            },
            'description': 'Novel pathogen with high transmission and mortality rates'
        },
        'CYBER_PEARL_HARBOR': {
            'category': MetricCategory.TECHNOLOGICAL,
            'metrics': {
                'cybersecurity': -0.8,
                'digital_economy_share': -0.4,
                'institutional_trust': -0.5
            },
            'description': 'Coordinated attack cripples critical infrastructure'
        }
    }
    
    effect = shock_effects.get(event_type)
    if not effect:
        return "Unknown event type"
    
    # Apply metric changes
    for metric, impact in effect['metrics'].items():
        current = metrics.get_metric(effect['category'], metric)
        new_value = np.clip(current + impact, 0, 1)
        metrics.set_metric(effect['category'], metric, new_value)
    
    return effect['description']

def compare_civilizations(engine: PsychohistoryEngine, civ_names: List[str]):
    """Generate comparative analysis visualization"""
    # Prepare data
    stability_scores = []
    risk_levels = []
    patterns_count = []
    
    for name in civ_names:
        analysis = engine.analyze_civilization(name)
        stability_scores.append(analysis['stability_score'])
        risk_levels.append(analysis['risk_level'])
        patterns_count.append(len(analysis['pattern_matches']))
    
    # Create figure with dark theme
    plt.figure(figsize=(14, 10), facecolor='#0f1a26')
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # Stability comparison
    ax1 = plt.subplot(gs[0, 0])
    ax1.set_facecolor('#0f1a26')
    bars = ax1.barh([n.replace('_', ' ') for n in civ_names], stability_scores, 
                   color=plt.cm.viridis(stability_scores))
    ax1.set_title('Civilization Stability Scores', color='#e0e0e0', fontsize=12)
    ax1.tick_params(axis='x', colors='#e0e0e0')
    ax1.tick_params(axis='y', colors='#e0e0e0')
    ax1.xaxis.label.set_color('#e0e0e0')
    ax1.set_xlim(0, 1.0)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', 
                ha='left', va='center', color='#e0e0e0')
    
    # Risk level comparison
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_facecolor('#0f1a26')
    risk_numeric = [{'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}[r] for r in risk_levels]
    colors = ['#2ca02c', '#ffd700', '#ff7f0e', '#d62728']
    bars = ax2.barh([n.replace('_', ' ') for n in civ_names], risk_numeric, 
                   color=[colors[min(int(r * 3), 3)] for r in risk_numeric])
    ax2.set_title('Risk Level Comparison', color='#e0e0e0', fontsize=12)
    ax2.tick_params(axis='x', colors='#e0e0e0')
    ax2.tick_params(axis='y', colors='#e0e0e0')
    ax2.set_xlim(0, 1.0)
    
    # Pattern detection comparison
    ax3 = plt.subplot(gs[1, :])
    ax3.set_facecolor('#0f1a26')
    
    # Prepare pattern data
    pattern_types = {}
    for name in civ_names:
        analysis = engine.analyze_civilization(name)
        for pattern in analysis['pattern_matches']:
            ptype = pattern['pattern_name']
            pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
    
    # Create radar chart
    categories = list(pattern_types.keys())
    values = [pattern_types[cat] for cat in categories]
    
    # Complete the circle
    values += values[:1]
    categories += categories[:1]
    
    N = len(categories) - 1
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax3 = plt.subplot(gs[1, :], polar=True)
    ax3.set_theta_offset(np.pi / 2)
    ax3.set_theta_direction(-1)
    
    # Draw one axe per variable
    plt.xticks(angles[:-1], categories, color='#e0e0e0', size=10)
    
    # Draw ylabels
    ax3.set_rlabel_position(0)
    plt.yticks([1, 2, 3], ["1", "2", "3"], color='#e0e0e0', size=8)
    plt.ylim(0, max(values) + 1)
    
    # Plot data
    ax3.plot(angles, values, linewidth=2, linestyle='solid', color='#3498db')
    ax3.fill(angles, values, '#3498db', alpha=0.2)
    
    ax3.set_title('Historical Pattern Detection Frequency', color='#e0e0e0', fontsize=12, pad=20)
    
    # Final styling
    plt.suptitle('Civilization Comparative Analysis', color='#e0e0e0', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def run_interactive_analysis():
    """Run an interactive analysis session with enhanced features"""
    print("\n🌐 Enhanced Psychohistory Interactive Analysis")
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
    
    # Store analysis history for comparison
    analysis_history = {}
    
    while True:
        print("\nMain Menu:")
        print("1. Analyze a civilization")
        print("2. Apply shock event to civilization")
        print("3. Compare civilizations")
        print("4. Generate report")
        print("0. Exit")
        
        try:
            choice = int(input("Enter choice: "))
            if choice == 0:
                break
            elif choice not in [1, 2, 3, 4]:
                print("Invalid choice, please try again.")
                continue
        except ValueError:
            print("Invalid input, please enter a number.")
            continue
        
        if choice == 1:  # Analyze civilization
            print("\nSelect a civilization to analyze:")
            for i, civ_name in enumerate(civilizations.keys(), 1):
                print(f"{i}. {civ_name.replace('_', ' ')}")
            print("0. Back")
            
            try:
                civ_choice = int(input("Enter choice: "))
                if civ_choice == 0:
                    continue
                civ_name = list(civilizations.keys())[civ_choice-1]
            except (ValueError, IndexError):
                print("Invalid choice, please try again.")
                continue
            
            # Perform analysis
            metrics = civilizations[civ_name]
            engine.add_civilization(civ_name, metrics)
            analysis = engine.analyze_civilization(civ_name)
            timeline = engine.predict_timeline(civ_name)
            
            # Store for comparison
            analysis_history[civ_name] = analysis
            
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
            for period in timeline:
                print(f"\n{period['timeframe']}:")
                for pred in period['predictions']:
                    print(f"- {pred['outcome']} (Confidence: {pred['confidence']:.0%})")
            
            # Show visualizations
            dashboard.display_analysis(analysis)
            display_timeline(timeline, civ_name)
            
            input("\nPress Enter to continue...")
            
        elif choice == 2:  # Apply shock event
            print("\nSelect civilization to shock:")
            for i, civ_name in enumerate(civilizations.keys(), 1):
                print(f"{i}. {civ_name.replace('_', ' ')}")
            print("0. Back")
            
            try:
                civ_choice = int(input("Enter choice: "))
                if civ_choice == 0:
                    continue
                civ_name = list(civilizations.keys())[civ_choice-1]
            except (ValueError, IndexError):
                print("Invalid choice, please try again.")
                continue
                
            print("\nSelect shock event type:")
            shocks = ['FINANCIAL_CRASH', 'AI_REVOLT', 'CLIMATE_DISASTER', 'PANDEMIC', 'CYBER_PEARL_HARBOR']
            for i, shock in enumerate(shocks, 1):
                print(f"{i}. {shock.replace('_', ' ')}")
            print("0. Back")
            
            try:
                shock_choice = int(input("Enter choice: "))
                if shock_choice == 0:
                    continue
                shock_type = shocks[shock_choice-1]
            except (ValueError, IndexError):
                print("Invalid choice, please try again.")
                continue
                
            # Apply shock
            description = simulate_shock_event(civilizations[civ_name], shock_type)
            print(f"\n⚡ Applied Shock: {description}")
            print(f"Civilization '{civ_name}' has been modified. Re-analyze to see impact.")
            
        elif choice == 3:  # Compare civilizations
            if len(analysis_history) < 2:
                print("\n⚠️ Need at least 2 civilizations analyzed for comparison")
                continue
                
            print("\nSelect civilizations to compare (choose 2-4):")
            valid_civs = list(analysis_history.keys())
            for i, civ_name in enumerate(valid_civs, 1):
                print(f"{i}. {civ_name.replace('_', ' ')}")
            print("0. Back")
            
            try:
                choices = input("Enter choices separated by commas: ")
                if choices == '0':
                    continue
                choices = [int(c.strip()) for c in choices.split(',')]
                selected_civs = [valid_civs[i-1] for i in choices if 1 <= i <= len(valid_civs)]
                
                if 2 <= len(selected_civs) <= 4:
                    compare_civilizations(engine, selected_civs)
                else:
                    print("Please select between 2 and 4 civilizations.")
            except (ValueError, IndexError):
                print("Invalid input, please try again.")
                
        elif choice == 4:  # Generate report
            generate_sample_report(engine)
            print("\n✅ Report generated: psychohistory_report.json")

def generate_sample_report(engine: PsychohistoryEngine):
    """Generate a comprehensive sample report"""
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
    
    return report

if __name__ == "__main__":
    print("🚀 Enhanced Psychohistory Demonstration System")
    print("=" * 60)
    print("Features:")
    print("- Timeline visualization")
    print("- Shock event simulation")
    print("- Comparative analysis")
    print("- 8 civilization types including modern crises")
    print("=" * 60)
    
    print("1. Interactive Analysis")
    print("2. Generate Sample Report")
    print("3. Exit")
    
    try:
        choice = int(input("Select mode: "))
        if choice == 1:
            run_interactive_analysis()
        elif choice == 2:
            engine = PsychohistoryEngine()
            generate_sample_report(engine)
            print("Report generated: psychohistory_report.json")
    except ValueError:
        print("Invalid input")
