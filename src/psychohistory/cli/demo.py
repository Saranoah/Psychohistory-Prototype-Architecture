#!/usr/bin/env python3
"""
Psychohistory Interactive Demo CLI
Command-line interface for civilizational analysis
"""

import sys
import argparse
from datetime import datetime
from typing import Dict, List

# Import from your core modules
from ..core.engine import PsychohistoryEngine
from ..core.metrics import CivilizationMetrics, MetricCategory

def create_sample_civilization(name: str) -> CivilizationMetrics:
    """Create a sample civilization for demonstration"""
    metrics = CivilizationMetrics()
    
    # Set some sample values based on name
    if name.lower() == "usa":
        metrics.update_metric(MetricCategory.ECONOMIC, 'wealth_inequality', 0.82)
        metrics.update_metric(MetricCategory.POLITICAL, 'institutional_trust', 0.35)
        metrics.update_metric(MetricCategory.SOCIAL, 'civic_engagement', 0.45)
    elif name.lower() == "example":
        metrics.update_metric(MetricCategory.ECONOMIC, 'wealth_inequality', 0.65)
        metrics.update_metric(MetricCategory.POLITICAL, 'institutional_trust', 0.55)
        metrics.update_metric(MetricCategory.SOCIAL, 'civic_engagement', 0.60)
    
    return metrics

def run_analysis(civ_name: str, verbose: bool = False):
    """Run analysis on a civilization"""
    engine = PsychohistoryEngine()
    metrics = create_sample_civilization(civ_name)
    
    engine.add_civilization(civ_name, metrics)
    result = engine.analyze_civilization(civ_name)
    
    print(f"\n=== Psychohistory Analysis: {civ_name} ===")
    print(f"Analysis Date: {result.date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Risk Score: {result.risk_score:.3f}/1.000")
    
    if result.pattern_matches:
        print(f"\nPattern Matches ({len(result.pattern_matches)}):")
        for match in result.pattern_matches[:3]:  # Show top 3
            print(f"  - {match['pattern_name']}: {match['match_score']:.1%} confidence")
            if verbose:
                print(f"    Outcome: {match['predicted_outcome']}")
                print(f"    Timeframe: {match['timeframe']}")
    
    if result.recommendations:
        print(f"\nTop Recommendations ({len(result.recommendations)}):")
        for i, rec in enumerate(result.recommendations[:5], 1):
            print(f"  {i}. {rec['action']}")
            if verbose:
                print(f"     Efficacy: {rec['efficacy']:.1%}, Category: {rec['category']}")

def list_metrics(civ_name: str):
    """List current metrics for a civilization"""
    metrics = create_sample_civilization(civ_name)
    
    print(f"\n=== Metrics for {civ_name} ===")
    for category in MetricCategory:
        print(f"\n{category.name}:")
        category_metrics = metrics.get_metrics(category)
        for name, value in category_metrics.items():
            print(f"  {name}: {value:.3f}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Psychohistory Civilizational Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  psychohistory-demo analyze USA
  psychohistory-demo analyze example --verbose
  psychohistory-demo metrics USA
  psychohistory-demo --version
        """
    )
    
    parser.add_argument('--version', action='version', version='psychohistory 2.0.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a civilization')
    analyze_parser.add_argument('civilization', help='Civilization name (usa, example, or custom)')
    analyze_parser.add_argument('-v', '--verbose', action='store_true', 
                              help='Show detailed output')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show civilization metrics')
    metrics_parser.add_argument('civilization', help='Civilization name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'analyze':
            run_analysis(args.civilization, args.verbose)
        elif args.command == 'metrics':
            list_metrics(args.civilization)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
