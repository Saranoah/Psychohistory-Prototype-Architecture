**CONTRIBUTING GUIDE**

```markdown
# Contributing to Psychohistory System

*Systematic analysis of civilizational patterns and risks*

## Setup Your Environment

```bash
git clone https://github.com/Saranoah/Psychohistory-Prototype-Architecture.git
cd Psychohistory-Prototype-Architecture
pip install -e .
pytest tests/ --cov=src/psychohistory
```

## Contribution Types

### Historical Analysis
- **Case Studies**: Add new civilizational analyses with:
  - Metric definitions and thresholds
  - Pattern validation against historical outcomes
  - Uncertainty quantification for predictions
  - Example: `case_studies/roman_republic_analysis.py`

### Code Improvements
- **Core Modules**: Enhance pattern matching, uncertainty calculation, or metric tracking
- **Analysis Tools**: Improve visualization, reporting, or intervention simulation
- **Testing**: Add historical validation tests or edge case handling

### Data Standards
Historical datasets should include:
```csv
civilization,period,metric_category,metric_name,value,confidence,source
Roman_Republic,100_BCE,POLITICAL,institutional_trust,0.3,0.8,Polybius_Histories
```

## Workflow

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/byzantine-analysis`
3. **Implement** with tests and documentation
4. **Validate**: Run test suite and check against historical examples
5. **Submit PR** with clear description of changes

## Code Standards

### Documentation
```python
def calculate_stability_score(metrics: Dict[str, float]) -> float:
    """
    Calculate composite stability score for a civilization.
    
    Args:
        metrics: Dictionary of normalized metric values (0-1)
        
    Returns:
        Stability score (0-1, where 1 is most stable)
        
    Historical Validation:
        - Roman Republic (100 BCE): Expected ~0.3 
        - Pax Romana (100 CE): Expected ~0.8
    """
```

### Testing Requirements
- Unit tests for all new functions
- Historical validation against known outcomes
- Edge case handling (missing data, extreme values)
- Performance benchmarks for large datasets

## Review Process

**Automated Checks:**
- Code quality (flake8, mypy)
- Test coverage (minimum 80%)
- Historical validation suite

**Human Review:**
- Code architecture and maintainability
- Historical accuracy and interpretation
- Statistical methodology validation

## Quick Start Ideas

**Beginner:**
- Add historical examples to existing patterns
- Improve error handling in core modules
- Create visualization for specific civilization

**Intermediate:** 
- Implement new historical pattern from literature
- Add uncertainty quantification for existing predictions
- Create comparative analysis between civilizations

**Advanced:**
- Design new metric categories or calculation methods
- Implement intervention simulation improvements
- Build automated pattern discovery algorithms

## Guidelines

- **Scientific Rigor**: All claims must be supported by historical evidence
- **Uncertainty Acknowledgment**: Always quantify and communicate prediction uncertainty  
- **Reproducible Analysis**: Provide clear methodology and data sources
- **Practical Focus**: Prioritize actionable insights over theoretical complexity

Questions? Open an issue with the `question` label.
```

