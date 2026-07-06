"""
Setup configuration for Maritime Domain Awareness System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="maritime-domain-awareness",
    version="1.0.0",
    author="Maritime Surveillance Team", 
    author_email="contact@maritime-surveillance.com",
    description="A comprehensive deep learning system for maritime surveillance",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/maritime-domain-awareness",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dashboard": [
            "streamlit>=1.20.0",
            "plotly>=5.0.0", 
            "seaborn>=0.11.0"
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950"
        ]
    },
    entry_points={
        "console_scripts": [
            "maritime-dashboard=run_dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
)