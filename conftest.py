"""
Put the project root on sys.path for pytest.

The package is run as `python -m ambient.main` from the repo root, so `config`
is a top-level module rather than part of the `ambient` package. Without this,
`import config` inside the tests fails depending on which directory you invoke
pytest from.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
