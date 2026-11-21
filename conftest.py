"""Make the repository root importable when pytest is run from anywhere."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)
