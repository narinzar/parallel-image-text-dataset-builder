"""Ensure the repo root is importable so `from src... import ...` works in tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
