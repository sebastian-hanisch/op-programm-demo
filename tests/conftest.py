"""Gemeinsame Einrichtung: der Wurzelordner steht im Importpfad (die Module der Demo liegen dort)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
