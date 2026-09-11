"""
Setup configuration for vitalguard package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# Core dependencies
install_requires = [
    "torch>=2.1.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "pyarrow>=14.0.0",
    "pyyaml>=6.0.1",
    "tqdm>=4.66.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.13.0",
    "statsmodels>=0.14.0",
    "lifelines>=0.27.0",
]

# Optional dependencies
extras_require = {
    "database": [
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
    ],
    "wandb": [
        "wandb>=0.16.0",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.10.0",
        "isort>=5.12.0",
        "flake8>=6.1.0",
        "mypy>=1.6.0",
        "pre-commit>=3.5.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.25.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
    ],
}

setup(
    name="vitalguard",
    version="0.1.0",
    description=(
        "Deep learning framework for clinical time-series anomaly detection "
        "and early warning in the Intensive Care Unit"
    ),
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/MJ-0324-2006/vitalguard",
    license="MIT",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "vitalguard-extract=scripts.extract_cohort:main",
            "vitalguard-train=scripts.train:main",
            "vitalguard-evaluate=scripts.evaluate:main",
            "vitalguard-monitor=scripts.simulate_monitoring:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords=[
        "clinical decision support", "anomaly detection", "early warning",
        "MIMIC", "ICU", "sepsis", "deep learning", "LSTM", "transformer", "TCN",
        "physiological signal processing", "time series"
    ],
    project_urls={
        "Bug Reports": "https://github.com/MJ-0324-2006/vitalguard/issues",
        "Source": "https://github.com/MJ-0324-2006/vitalguard",
        "Documentation": "https://github.com/MJ-0324-2006/vitalguard/blob/main/README.md",
    },
)

