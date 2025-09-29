#!/usr/bin/env python3
"""
Real-Time Computational Psychohistory v2.0 
Robust setup.py: safe defaults, dynamic version discovery, extras aggregated.
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# ---------- Configuration ----------
PACKAGE_NAME = "psychohistory"
ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
PKG_DIR = SRC_DIR / PACKAGE_NAME
README = ROOT_DIR / "README.md"
REQUIREMENTS = ROOT_DIR / "requirements.txt"

# Files to inspect for version
VERSION_PY = PKG_DIR / "__version__.py"
INIT_PY = PKG_DIR / "__init__.py"

# ---------- Helper functions ----------
def read_text_safe(path: Path, encoding="utf-8"):
    try:
        return path.read_text(encoding=encoding)
    except Exception:
        return None

def get_version():
    """Try multiple places for __version__ and fallback to '2.0.0'."""
    for p in (VERSION_PY, INIT_PY):
        text = read_text_safe(p)
        if text:
            m = re.search(r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", text, re.M)
            if m:
                return m.group(1)
    # Last resort fallback:
    return "2.0.0"

def get_long_description():
    txt = read_text_safe(README)
    if txt:
        return txt
    return "Real-Time Computational Psychohistory v2.0: AI-Aware Civilizational Monitoring Framework."

def get_requirements():
    """Read requirements.txt (lowercase). Return reasonable fallback if missing."""
    content = read_text_safe(REQUIREMENTS)
    if content:
        reqs = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith(("#", "-"))
        ]
        return reqs
    # fallback minimal runtime deps
    return [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ]

# ---------- Extras definition (programmatic so 'full' is union) ----------
extras = {
    "dev": [
        "pytest>=7.0",
        "pytest-asyncio>=0.21",
        "pytest-cov>=4.0",
        "black>=23.7.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "isort>=5.12.0",
        "pre-commit>=3.3.3",
    ],
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
    ],
    "ml": [
        "torch>=2.0.1",
        "transformers>=4.30.2",
        "scikit-learn>=1.3.0",
    ],
    "viz": [
        "dash>=2.11.0",
        "plotly>=5.15.0",
        "streamlit>=1.25.0",
    ],
    "analysis": [
        "scipy>=1.7.0",
        "networkx>=2.6.0",
    ],
}

# Compute 'full' as union of all extras (avoid self-referential extras)
_all = set()
for k, v in extras.items():
    _all.update(v)
extras["full"] = sorted(_all)

# ---------- Metadata ----------
metadata = {
    "name": PACKAGE_NAME,
    "version": get_version(),
    "author": "Israa Ali",
    "author_email": "israali2019@yahoo.com",
    "description": "Real-Time Computational Psychohistory v2.0: AI-Aware Civilizational Monitoring Framework",
    "long_description": get_long_description(),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/Saranoah/Psychohistory-Prototype-Architecture",
    "project_urls": {
        "Bug Tracker": "https://github.com/Saranoah/Psychohistory-Prototype-Architecture/issues",
        "Source": "https://github.com/Saranoah/Psychohistory-Prototype-Architecture",
        "Documentation": "https://github.com/Saranoah/Psychohistory-Prototype-Architecture/blob/main/docs/index.md",
    },
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    "package_dir": {"": "src"},
    "packages": find_packages(where="src", include=[f"{PACKAGE_NAME}*"]),
    "python_requires": ">=3.8",
    "install_requires": get_requirements(),
    "extras_require": extras,

    # ----------- FIXED ENTRY POINTS -----------
    "entry_points": {
        "console_scripts": [
            # Main CLI command for psychohistory
            "psychohistory=psychohistory.cli:cli",
        ],
    },
    # ------------------------------------------

    "include_package_data": True,
    "package_data": {
        PACKAGE_NAME: [
            "data/*.json",
            "data/patterns/*.yaml",
            "templates/*.html",
            "py.typed",
        ]
    },
    # keep minimal data_files if you want to install docs to system locations (optional)
    "data_files": [("share/docs/psychohistory", ["README.md", "LICENSE"])] if (ROOT_DIR / "README.md").exists() else [],
    "zip_safe": False,
    "keywords": [
        "psychohistory",
        "civilization",
        "forecasting",
        "AI",
        "social dynamics",
    ],
    "license": "MIT",
}

# ---------- Run setup ----------
if __name__ == "__main__":
    setup(**metadata)
