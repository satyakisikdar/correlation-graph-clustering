"""
General utilities file
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    basedir: Path = Path('../')
    data_dir: Path = basedir / 'data'
    results_dir: Path = basedir / 'results'
    