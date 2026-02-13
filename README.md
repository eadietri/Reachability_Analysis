# Reachability Analysis

A Python project implementing data-driven reachability analysis methods.

Reachability analysis is an important method in providing safety guarantees for systems with unknown or uncertain dynamics.

## Overview

This project contains implementations for various approaches to data-driven reachability using *probabilistic guarantees* and *scenario-based methods*.

## Repository Structure

```
Reachability_Analysis/
│
├── src/
│   └── reachability_analysis/
│       ├── __init__.py
│       ├── probabilistic_guarantees/
│       │   ├── __init__.py
│       │   └── binomial_utils.py # Calculate binomial tail inversion to get bound for Holdout Method.
│       ├── scenario_approaches/
│       │   ├── ellipsoids/
│       │       └── __init__.py
|       |       └── ellipsoid_binomial.py # Calculates epsilon for ellipsoidal reachable set/tube using binomial tail inversion. 
|       |       └── ellipsoid_utils.py # Approximate reachability problem using ellipsoids.
|       |       └── plotting_utils.py # Plot ellipsoids for visualization purposes.
|       |   ├── zonotopes/
|       |       └── __init__.py
|       |       └── zonotope_utils.py # Approximate reachability problem using zonotopes.
│
├── README.md
├── pyproject.toml
└── requirements.txt 
```

## Installation
<pre> ``` git clone https://github.com/eadietri/Reachability_Analysis.git 
          cd Reachability_Analysis 
          python -m venv venv
          source venv/bin/activate
          pip install -e .``` </pre>

