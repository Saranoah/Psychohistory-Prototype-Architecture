#!/usr/bin/env python3
"""
Real-Time Computational Psychohistory
Advanced setup configuration for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Constants
PACKAGE_NAME = "psychohistory"
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"
README = ROOT_DIR / "README.md"
REQUIREMENTS = ROOT_DIR / "requirements.txt"
VERSION_FILE = SRC_DIR / PACKAGE_NAME / "__version__.py"

def get_version():
    """Extract version from package __version__.py"""
    version_content = VERSION_FILE.read_text()
    version_match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", 
        version_content, 
        re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

def get_long_description():
    """Read long description from README"""
    return README.read_text(encoding="utf-8")

def get_requirements():
    """Parse requirements file"""
    return [
        line.strip() 
        for line in REQUIREMENTS.read_text().splitlines() 
        if line.strip() and not line.startswith(("#", "-"))
    ]

# Package metadata
metadata = {
    "name": PACKAGE_NAME,
    "version": get_version(),
    "author": "Your Name",
    "author_email": "your.email@domain.com",
    "description": (
        "Real-Time Computational Psychohistory: "
        "AI-Aware Civilizational Monitoring Framework"
    ),
    "long_description": get_long_description(),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/yourusername/psychohistory",
    "project_urls": {
        "Bug Tracker": "https://github.com/yourusername/psychohistory/issues",
        "Documentation": "https://psychohistory.readthedocs.io",
        "Source Code": "https://github.com/yourusername/psychohistory",
        "Changelog": "https://github.com/yourusername/psychohistory/releases",
    },
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Sociology",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Framework :: AsyncIO",
        "Typing :: Typed",
    ],
    "package_dir": {"": "src"},
    "packages": find_packages(
        where="src",
        include=["psychohistory*"],
        exclude=["tests*", "docs*", "examples*"]
    ),
    "python_requires": ">=3.8",
    "install_requires": get_requirements(),
    "extras_require": {
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.10.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.3",
        ],
        "docs": [
            "sphinx>=7.0.1",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.23.0",
            "myst-parser>=2.0.0",
            "sphinx-copybutton>=0.5.2",
        ],
        "ml": [
            "tensorflow>=2.12.0",
            "torch>=2.0.1",
            "transformers>=4.30.2",
            "scikit-learn>=1.3.0",
            "xgboost>=1.7.5",
        ],
        "viz": [
            "dash>=2.11.0",
            "plotly>=5.15.0",
            "jupyter>=1.0.0",
            "seaborn>=0.12.2",
            "bokeh>=3.2.1",
        ],
        "full": [
            "psychohistory[ml,viz,dev,docs]"
        ]
    },
    "entry_points": {
        "console_scripts": [
            "psychohistory-monitor=psychohistory.cli.monitor:main",
            "psychohistory-analyze=psychohistory.cli.analyze:main",
            "psychohistory-dashboard=psychohistory.cli.dashboard:main",
            "psychohistory-train=psychohistory.cli.train:main [ml]",
        ],
    },
    "include_package_data": True,
    "package_data": {
        PACKAGE_NAME: [
            "data/*.json",
            "data/patterns/*.yaml",
            "data/configs/*.yml",
            "templates/*.html",
            "templates/*.jinja2",
            "py.typed",  # PEP 561 type marker
        ],
    },
    "data_files": [
        ("share/docs/psychohistory", ["README.md", "LICENSE", "CONTRIBUTING.md"]),
    ],
    "zip_safe": False,
    "keywords": [
        "psychohistory",
        "civilization",
        "prediction",
        "artificial intelligence",
        "historical patterns",
        "social dynamics",
        "complex systems",
        "cliodynamics",
        "forecasting",
        "risk assessment",
        "computational sociology",
    ],
    "license": "MIT",
    "platforms": ["any"],
    "options": {
        "bdist_wheel": {
            "universal": True
        }
    },
}

if __name__ == "__main__":
    setup(**metadata)
