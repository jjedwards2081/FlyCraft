"""Paths, and the import hook that makes the fly-brain repo's modules importable."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLY_BRAIN = ROOT / 'fly-brain'
ANNOTATIONS = ROOT / 'data' / 'flywire_neuron_annotations.tsv'
CALIBRATION = ROOT / 'data' / 'calibration.json'

# fly-brain is used unmodified; its code/ folder holds benchmark.py and run_pytorch.py
sys.path.insert(0, str(FLY_BRAIN / 'code'))
