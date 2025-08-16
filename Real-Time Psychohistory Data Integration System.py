# =====================
# ENHANCED COSMOLOGICAL ANALYSIS ENGINE
# =====================
class CosmologicalImpactAnalyzer:
    """Analyzes the societal impact of cosmological discoveries"""
    
    DISCOVERY_IMPACT_FACTORS = {
        'dark_matter_research': {
            'physics': 1.8,
            'technology': 1.5,
            'philosophy': 2.0,
            'economy': 1.2
        },
        'dark_energy_research': {
            'physics': 2.0,
            'technology': 1.7,
            'philosophy': 2.5,
            'economy': 1.3
        },
        'gravitational_wave_discoveries': {
            'physics': 1.5,
            'technology': 1.2,
            'philosophy': 1.3,
            'economy': 0.8
        },
        'exoplanet_discoveries': {
            'physics': 1.2,
            'technology': 1.6,
            'philosophy': 1.8,
            'economy': 1.4
        },
        'cosmic_ray_anomalies': {
            'physics': 1.7,
            'technology': 1.3,
            'philosophy': 1.5,
            'economy': 0.9
        },
        'multiverse_evidence': {
            'physics': 2.5,
            'technology': 1.8,
            'philosophy': 3.0,
            'economy': 1.2
        }
    }
    
    def __init__(self):
        self.cumulative_impact = 0.0
        self.recent_breakthroughs = []
    
    def analyze_discoveries(self, discoveries: List[Dict]) -> Dict:
        """Analyze a set of cosmological discoveries"""
        impacts = []
        total_impact = 0.0
        
        for discovery in discoveries:
            if discovery['value'] > 0.6:  # Significant discovery
                impact = self._calculate_discovery_impact(discovery)
                impacts.append({
                    'discovery': discovery['subcategory'],
                    'value': discovery['value'],
                    'impact': impact,
                    'societal_effects': self._get_societal_effects(discovery['metric_name'])
                })
                total_impact += impact
                self.recent_breakthroughs.append(discovery)
        
        # Calculate cumulative impact with decay
        self.cumulative_impact = (self.cumulative_impact * 0.9) + (total_impact * 0.5)
        
        return {
            'total_impact': total_impact,
            'cumulative_impact': self.cumulative_impact,
            'discovery_impacts': impacts,
            'paradigm_shift_risk': self._calculate_paradigm_shift_risk()
        }
    
    def _calculate_discovery_impact(self, discovery: Dict) -> float:
        """Calculate impact score for a discovery"""
        base_value = discovery['value']
        metric = discovery['metric_name']
        
        # Base impact factors
        impact_factors = self.DISCOVERY_IMPACT_FACTORS.get(metric, {})
        impact = base_value * sum(impact_factors.values()) / len(impact_factors)
        
        # Breakthrough multiplier
        if base_value > 0.8:
            impact *= 1.8  # Major breakthrough
        elif base_value > 0.7:
            impact *= 1.4  # Significant breakthrough
            
        return min(1.0, impact)
    
    def _get_societal_effects(self, metric_name: str) -> List[str]:
        """Get potential societal effects of a discovery type"""
        effects_map = {
            'dark_matter_research': [
                "New physics models",
                "Advanced materials research",
                "Philosophical debates about matter"
            ],
            'dark_energy_research': [
                "Energy technology possibilities",
                "Cosmology paradigm shifts",
                "Religious reinterpretations"
            ],
            'gravitational_wave_discoveries': [
                "New astronomy capabilities",
                "Space-time technology concepts",
                "Educational curriculum updates"
            ],
            'exoplanet_discoveries': [
                "Increased space exploration funding",
                "Public interest in astronomy",
                "Philosophical questions about life"
            ],
            'cosmic_ray_anomalies': [
                "New physics research directions",
                "Radiation protection technology",
                "Theoretical model revisions"
            ],
            'multiverse_evidence': [
                "Philosophical upheaval",
                "Religious reinterpretations",
                "Science fiction becomes reality",
                "Educational system challenges"
            ]
        }
        return effects_map.get(metric_name, [
            "Scientific paradigm shifts",
            "Educational system updates",
            "Philosophical reconsiderations"
        ])
    
    def _calculate_paradigm_shift_risk(self) -> float:
        """Calculate risk of major paradigm shift"""
        if not self.recent_breakthroughs:
            return 0.0
            
        # Count significant breakthroughs in last month
        recent_count = sum(1 for d in self.recent_breakthroughs 
                          if d['value'] > 0.7 and 
                          (datetime.utcnow() - d['timestamp']).days < 30)
        
        # Calculate risk based on breakthrough frequency and impact
        risk = min(1.0, recent_count * 0.2 + self.cumulative_impact * 0.5)
        
        # Reset breakthroughs older than 60 days
        self.recent_breakthroughs = [
            d for d in self.recent_breakthroughs 
            if (datetime.utcnow() - d['timestamp']).days < 60
        ]
        
        return risk

# =====================
# ENHANCED SCIENTIFIC IMPACT ANALYZER
# =====================
class ScientificImpactAnalyzer:
    """Analyzes the societal impact of scientific breakthroughs"""
    
    FIELD_DISRUPTION_FACTORS = {
        'physics': {
            'technology': 2.0,
            'economy': 1.8,
            'social': 1.5,
            'political': 1.2
        },
        'biology': {
            'healthcare': 2.2,
            'ethics': 1.8,
            'social': 1.7,
            'economy': 1.5
        },
        'mathematics': {
            'technology': 1.8,
            'security': 2.0,
            'economy': 1.6,
            'education': 1.4
        },
        'space': {
            'economy': 2.0,
            'geopolitics': 1.8,
            'technology': 1.7,
            'social': 1.4
        }
    }
    
    def __init__(self):
        self.disruption_wave_level = 0.0
    
    def analyze_breakthroughs(self, breakthroughs: List[Dict]) -> Dict:
        """Analyze a set of scientific breakthroughs"""
        disruptions = []
        total_disruption = 0.0
        
        for breakthrough in breakthroughs:
            if breakthrough['value'] > 0.6:  # Significant breakthrough
                disruption = self._calculate_disruption(breakthrough)
                disruptions.append({
                    'field': breakthrough['metric_name'],
                    'category': breakthrough['subcategory'],
                    'value': breakthrough['value'],
                    'disruption': disruption,
                    'effects': self._get_disruption_effects(breakthrough)
                })
                total_disruption += disruption
        
        # Update disruption wave level
        self.disruption_wave_level = min(1.0, 
            (self.disruption_wave_level * 0.85) + (total_disruption * 0.3))
        
        return {
            'total_disruption': total_disruption,
            'disruption_wave': self.disruption_wave_level,
            'breakthrough_disruptions': disruptions,
            'transformation_potential': self._calculate_transformation_potential()
        }
    
    def _calculate_disruption(self, breakthrough: Dict) -> float:
        """Calculate disruption score for a breakthrough"""
        base_value = breakthrough['value']
        field_type = breakthrough['subcategory']
        
        # Base disruption factors
        disruption_factors = self.FIELD_DISRUPTION_FACTORS.get(field_type, {})
        disruption = base_value * sum(disruption_factors.values()) / len(disruption_factors)
        
        # Breakthrough multiplier
        if base_value > 0.8:
            disruption *= 1.7  # Revolutionary breakthrough
        elif base_value > 0.7:
            disruption *= 1.4  # Major breakthrough
            
        return min(1.0, disruption)
    
    def _get_disruption_effects(self, breakthrough: Dict) -> List[str]:
        """Get potential disruption effects"""
        field = breakthrough['subcategory']
        metric = breakthrough['metric_name']
        
        effects_map = {
            'physics': {
                'quantum_computing_progress': [
                    "Cryptography revolution",
                    "Drug discovery acceleration",
                    "AI capabilities leap"
                ],
                'fusion_energy_progress': [
                    "Energy industry disruption",
                    "Geopolitical power shifts",
                    "Environmental impact reduction"
                ],
                'room_temp_superconductors': [
                    "Energy transmission revolution",
                    "Transportation transformation",
                    "Electronic device revolution"
                ]
            },
            'biology': {
                'longevity_research': [
                    "Healthcare system strain",
                    "Retirement age redefinition",
                    "Population dynamics shift"
                ],
                'consciousness_understanding': [
                    "AI ethics debates",
                    "Legal system challenges",
                    "Philosophical reconsiderations"
                ],
                'genetic_engineering': [
                    "Medical breakthroughs",
                    "Ethical dilemmas",
                    "Agricultural revolution"
                ]
            }
        }
        
        # Return field-specific effects if available, otherwise general effects
        field_effects = effects_map.get(field, {})
        specific_effects = field_effects.get(metric, [])
        
        if specific_effects:
            return specific_effects
        
        # General effects by field
        return {
            'physics': [
                "Industrial transformation",
                "Technology paradigm shifts",
                "New economic sectors"
            ],
            'biology': [
                "Healthcare revolution",
                "Ethical debates",
                "Lifestyle changes"
            ],
            'mathematics': [
                "Computing revolution",
                "Security vulnerabilities",
                "Optimization breakthroughs"
            ],
            'space': [
                "New economic frontiers",
                "Geopolitical competition",
                "Resource availability changes"
            ]
        }.get(field, [
            "Technological disruption",
            "Economic restructuring",
            "Social adaptation required"
        ])
    
    def _calculate_transformation_potential(self) -> float:
        """Calculate potential for societal transformation"""
        return min(1.0, self.disruption_wave_level * 1.2)

# =====================
# UPDATED PSYCHOHISTORY ENGINE
# =====================
class EnhancedPsychohistoryEngine(PsychohistoryEngine):
    """Enhanced engine with cosmological and scientific impact analysis"""
    
    def __init__(self):
        super().__init__()
        self.cosmo_analyzer = CosmologicalImpactAnalyzer()
        self.science_analyzer = ScientificImpactAnalyzer()
        
        # Update weights to include new factors
        self.category_weights = {
            'economic': 0.22,
            'political': 0.22,
            'ai_usage': 0.18,
            'cosmological': 0.18,
            'scientific': 0.20
        }
    
    def calculate_stability_score(self, hours_back: int = 24) -> float:
        """Enhanced stability calculation with discovery impacts"""
        stability = super().calculate_stability_score(hours_back)
        
        # Get recent discoveries
        cosmo_data = self.db.get_category_data('cosmological', days_back=1)
        science_data = self.db.get_category_data('scientific', days_back=1)
        
        # Analyze impacts
        cosmo_impact = self.cosmo_analyzer.analyze_discoveries(cosmo_data)
        science_impact = self.science_analyzer.analyze_breakthroughs(science_data)
        
        # Adjust stability based on impacts
        stability -= cosmo_impact.get('paradigm_shift_risk', 0) * 0.15
        stability += science_impact.get('transformation_potential', 0) * 0.1
        stability -= science_impact.get('disruption_wave', 0) * 0.12
        
        return max(0.0, min(1.0, stability))
    
    def detect_patterns(self) -> List[Dict]:
        """Enhanced pattern detection with discovery analysis"""
        patterns = super().detect_patterns()
        
        # Add cosmological patterns
        cosmo_data = self.db.get_category_data('cosmological', days_back=14)
        cosmo_impact = self.cosmo_analyzer.analyze_discoveries(cosmo_data)
        
        if cosmo_impact['total_impact'] > 0.3:
            patterns.append({
                'detected': True,
                'name': 'Cosmological Paradigm Shift',
                'significance': min(1.0, cosmo_impact['total_impact'] * 1.2),
                'description': 'Significant cosmological discoveries challenging fundamental understanding',
                'implications': [
                    'Physics education overhaul needed',
                    'Technological innovation opportunities',
                    'Philosophical and religious reconsiderations',
                    'Space program acceleration'
                ]
            })
        
        # Add scientific disruption patterns
        science_data = self.db.get_category_data('scientific', days_back=14)
        science_impact = self.science_analyzer.analyze_breakthroughs(science_data)
        
        if science_impact['disruption_wave'] > 0.4:
            patterns.append({
                'detected': True,
                'name': 'Scientific Disruption Wave',
                'significance': min(1.0, science_impact['disruption_wave'] * 1.3),
                'description': 'Cluster of scientific breakthroughs causing societal disruption',
                'implications': [
                    'Regulatory framework challenges',
                    'Economic sector realignment',
                    'Workforce reskilling needs',
                    'Ethical and governance debates'
                ]
            })
        
        return patterns
    
    def get_dashboard_data(self) -> Dict:
        """Enhanced dashboard data with discovery metrics"""
        data = super().get_dashboard_data()
        
        # Add cosmological impact data
        cosmo_data = self.db.get_category_data('cosmological', days_back=30)
        data['cosmological_impact'] = self.cosmo_analyzer.analyze_discoveries(cosmo_data)
        
        # Add scientific disruption data
        science_data = self.db.get_category_data('scientific', days_back=30)
        data['scientific_disruption'] = self.science_analyzer.analyze_breakthroughs(science_data)
        
        # Add combined impact metric
        data['discovery_impact_index'] = (
            data['cosmological_impact']['cumulative_impact'] * 0.4 +
            data['scientific_disruption']['disruption_wave'] * 0.6
        )
        
        return data

# =====================
# ENHANCED WEB INTERFACE
# =====================
class EnhancedWebInterface(WebInterface):
    """Web interface with discovery visualization"""
    
    def create_dashboard_html(self) -> str:
        """Enhanced dashboard with discovery sections"""
        html = super().create_dashboard_html()
        
        # Get discovery data
        dashboard_data = self.engine.get_dashboard_data()
        cosmo_impact = dashboard_data.get('cosmological_impact', {})
        science_disruption = dashboard_data.get('scientific_disruption', {})
        
        # Add cosmological impact section
        cosmo_html = self._generate_cosmological_html(cosmo_impact)
        
        # Add scientific disruption section
        science_html = self._generate_scientific_html(science_disruption)
        
        # Insert new sections before the footer
        insertion_point = html.find('<div class="card">\n                    <h2>📊 Data Sources</h2>')
        if insertion_point != -1:
            new_html = (
                html[:insertion_point] +
                cosmo_html +
                science_html +
                html[insertion_point:]
            )
            return new_html
        
        return html
    
    def _generate_cosmological_html(self, impact_data: Dict) -> str:
        """Generate HTML for cosmological impact section"""
        if not impact_data:
            return ""
            
        impact_value = impact_data.get('cumulative_impact', 0)
        risk_value = impact_data.get('paradigm_shift_risk', 0)
        
        # Determine impact level and color
        if impact_value > 0.7:
            impact_level = "REVOLUTIONARY"
            impact_color = "#ff4444"
        elif impact_value > 0.5:
            impact_level = "HIGH"
            impact_color = "#ff8800"
        elif impact_value > 0.3:
            impact_level = "MODERATE"
            impact_color = "#ffaa00"
        else:
            impact_level = "LOW"
            impact_color = "#88dd00"
        
        # Determine risk level and color
        if risk_value > 0.6:
            risk_level = "CRITICAL"
            risk_color = "#ff4444"
        elif risk_value > 0.4:
            risk_level = "HIGH"
            risk_color = "#ff8800"
        elif risk_value > 0.2:
            risk_level = "MODERATE"
            risk_color = "#ffaa00"
        else:
            risk_level = "LOW"
            risk_color = "#88dd00"
        
        # Generate discoveries HTML
        discoveries_html = ""
        for discovery in impact_data.get('discovery_impacts', [])[:3]:
            discoveries_html += f"""
            <div style="margin: 15px 0; padding: 12px; border-left: 3px solid {impact_color}; background: #f9f9f9;">
                <h4 style="margin: 0 0 5px 0;">{discovery['discovery'].title()}</h4>
                <p><strong>Discovery Significance:</strong> {discovery['value']:.2f}</p>
                <p><strong>Societal Impact:</strong> {discovery['impact']:.2f}</p>
                <details>
                    <summary style="cursor: pointer; font-weight: bold;">Potential Effects</summary>
                    <ul>
                        {"".join(f"<li>{effect}</li>" for effect in discovery.get('societal_effects', []))}
                    </ul>
                </details>
            </div>
            """
        
        return f"""
        <div class="card">
            <h2>🌌 Cosmological Impact Analysis</h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0;">
                <div class="metric-card" style="border-color: {impact_color};">
                    <div class="metric-label" style="color: {impact_color};">Cumulative Impact</div>
                    <div class="metric-value" style="color: {impact_color};">{impact_value:.3f}</div>
                    <div style="color: {impact_color}; font-weight: bold;">{impact_level} IMPACT</div>
                </div>
                
                <div class="metric-card" style="border-color: {risk_color};">
                    <div class="metric-label" style="color: {risk_color};">Paradigm Shift Risk</div>
                    <div class="metric-value" style="color: {risk_color};">{risk_value:.3f}</div>
                    <div style="color: {risk_color}; font-weight: bold;">{risk_level} RISK</div>
                </div>
            </div>
            
            <h3>Recent Significant Discoveries</h3>
            {discoveries_html if discoveries_html else "<p>No significant recent discoveries</p>"}
        </div>
        """
    
    def _generate_scientific_html(self, disruption_data: Dict) -> str:
        """Generate HTML for scientific disruption section"""
        if not disruption_data:
            return ""
            
        disruption_value = disruption_data.get('disruption_wave', 0)
        transformation_value = disruption_data.get('transformation_potential', 0)
        
        # Determine disruption level and color
        if disruption_value > 0.7:
            disruption_level = "EXTREME"
            disruption_color = "#ff4444"
        elif disruption_value > 0.5:
            disruption_level = "HIGH"
            disruption_color = "#ff8800"
        elif disruption_value > 0.3:
            disruption_level = "MODERATE"
            disruption_color = "#ffaa00"
        else:
            disruption_level = "LOW"
            disruption_color = "#88dd00"
        
        # Determine transformation level and color
        if transformation_value > 0.7:
            transformation_level = "REVOLUTIONARY"
            transformation_color = "#00dd00"
        elif transformation_value > 0.5:
            transformation_level = "HIGH"
            transformation_color = "#88dd00"
        elif transformation_value > 0.3:
            transformation_level = "MODERATE"
            transformation_color = "#ffaa00"
        else:
            transformation_level = "LOW"
            transformation_color = "#ff8800"
        
        # Generate breakthroughs HTML
        breakthroughs_html = ""
        for breakthrough in disruption_data.get('breakthrough_disruptions', [])[:3]:
            breakthroughs_html += f"""
            <div style="margin: 15px 0; padding: 12px; border-left: 3px solid {disruption_color}; background: #f9f9f9;">
                <h4 style="margin: 0 0 5px 0;">{breakthrough['field'].replace('_', ' ').title()}</h4>
                <p><strong>Breakthrough Level:</strong> {breakthrough['value']:.2f}</p>
                <p><strong>Disruption Potential:</strong> {breakthrough['disruption']:.2f}</p>
                <details>
                    <summary style="cursor: pointer; font-weight: bold;">Potential Effects</summary>
                    <ul>
                        {"".join(f"<li>{effect}</li>" for effect in breakthrough.get('effects', []))}
                    </ul>
                </details>
            </div>
            """
        
        return f"""
        <div class="card">
            <h2>🔬 Scientific Disruption Analysis</h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0;">
                <div class="metric-card" style="border-color: {disruption_color};">
                    <div class="metric-label" style="color: {disruption_color};">Disruption Wave</div>
                    <div class="metric-value" style="color: {disruption_color};">{disruption_value:.3f}</div>
                    <div style="color: {disruption_color}; font-weight: bold;">{disruption_level} DISRUPTION</div>
                </div>
                
                <div class="metric-card" style="border-color: {transformation_color};">
                    <div class="metric-label" style="color: {transformation_color};">Transformation Potential</div>
                    <div class="metric-value" style="color: {transformation_color};">{transformation_value:.3f}</div>
                    <div style="color: {transformation_color}; font-weight: bold;">{transformation_level} POTENTIAL</div>
                </div>
            </div>
            
            <h3>Recent High-Impact Breakthroughs</h3>
            {breakthroughs_html if breakthroughs_html else "<p>No significant recent breakthroughs</p>"}
        </div>
        """

# =====================
# UPDATED MAIN SYSTEM
# =====================
class EnhancedPsychohistorySystem(PsychohistorySystem):
    """Enhanced system with discovery analysis"""
    
    def __init__(self):
        self.engine = EnhancedPsychohistoryEngine()
        self.web_interface = EnhancedWebInterface(self.engine)
        self.is_running = False

# =====================
# UPDATE MAIN FUNCTION
# =====================
async def main():
    """Main entry point with enhanced system"""
    print_banner()
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Psychohistory Monitor Demo')
    parser.add_argument('--host', default='localhost', help='Web server host (default: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Web server port (default: 5000)')
    parser.add_argument('--interval', type=int, default=30, help='Data collection interval in minutes (default: 30)')
    parser.add_argument('--test', action='store_true', help='Run quick test and exit')
    
    args = parser.parse_args()
    
    # Create enhanced system
    system = EnhancedPsychohistorySystem()
    
    if args.test:
        logger.info("🧪 Running enhanced test...")
        await system.initialize_demo_data()
        
        stability = system.engine.calculate_stability_score()
        patterns = system.engine.detect_patterns()
        
        print(f"\n✅ Enhanced Test Results:")
        print(f"   Stability Score: {stability:.3f}")
        print(f"   Patterns Detected: {len(patterns)}")
        
        # Discovery analysis
        cosmo_data = system.engine.db.get_category_data('cosmological', days_back=7)
        cosmo_impact = system.engine.cosmo_analyzer.analyze_discoveries(cosmo_data)
        print(f"   Cosmological Impact: {cosmo_impact['total_impact']:.3f}")
        
        science_data = system.engine.db.get_category_data('scientific', days_back=7)
        science_impact = system.engine.science_analyzer.analyze_breakthroughs(science_data)
        print(f"   Scientific Disruption: {science_impact['disruption_wave']:.3f}")
        
        print("\n🎯 Enhanced test completed successfully!")
        return
    
    # Run full enhanced system
    print(f"🚀 Starting enhanced system...")
    print(f"   Web Interface: http://{args.host}:{args.port}")
    print(f"   Collection Interval: {args.interval} minutes")
    print(f"   Press Ctrl+C to stop")
    print()
    
    try:
        await system.run(
            host=args.host, 
            port=args.port, 
            collection_interval=args.interval
        )
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutdown complete")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)
